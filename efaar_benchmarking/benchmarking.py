from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from geomloss import SamplesLoss
from joblib import Parallel, delayed
from scipy.stats import hypergeom, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import Bunch
from torch import from_numpy

import efaar_benchmarking.constants as cst
from efaar_benchmarking.benchmarking import (
    AggregateBy,
    AverageType,
    BenchmarkConfig,
    compound_gene_benchmark,
    compute_metrics,
    compute_similarities,
    process_predictions,
    sample_for_item,
)


class AverageType(Enum):
    MICRO = "micro"
    MACRO = "macro"


class AggregateBy(Enum):
    COMPOUND = "compound"
    GENE = "gene"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark computation."""

    average_type: AverageType = AverageType.MACRO
    aggregate_by: AggregateBy = AggregateBy.COMPOUND
    min_negatives: int = 20
    n_baseline_sims: int = 100
    random_seed: int = 42


def pert_signal_consistency_metric(
    arr: np.ndarray, sorted_null: np.ndarray = np.array([])
) -> float | None | tuple[float | None, float | None]:
    """
    Calculate the perturbation signal consistency metric, i.e., average cosine and associated p-value,
        for a given array.

    Args:
        arr (numpy.ndarray): The input array.
        sorted_null (numpy.ndarray, optional): Null distribution of the metric. Defaults to an empty array.
            If not empty, required to be sorted in ascending order prior to passing to this function.

    Returns:
        Union[Optional[float], tuple[Optional[float], Optional[float]]]:
        - If null is empty, returns the average cosine as a float. If the length of the input array is less than 2,
            returns None.
        - If null is not empty, returns a tuple containing the average cosine and p-value of the metric. If the length
            of the input array is less than 2, returns (None, None).
    """
    if len(arr) < 2:
        return np.nan if len(sorted_null) == 0 else (np.nan, np.nan)

    cosine_sim = np.clip(cosine_similarity(arr), -1, 1)  # to avoid floating point precision errors
    cosine_sim = cosine_sim[np.tril_indices(cosine_sim.shape[0], k=-1)].mean()

    if len(sorted_null) == 0:
        return cosine_sim
    else:
        pval = 1 - np.searchsorted(sorted_null, cosine_sim) / len(sorted_null)
        return cosine_sim, pval


def pert_signal_consistency_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    neg_ctrl_perts: list,
    keys_to_drop: list = [],
    n_jobs: int = 5,
) -> pd.DataFrame:
    """
    Perform perturbation consistency benchmarking on the given features and metadata.
    Filter out perturbations specified in the `keys_to_drop` list.
    Use negative control perturbations specified in the `neg_ctrl_perts` list for the null distribution.
    Calculate the query metrics for each perturbation and return them in a dataframe.

    Args:
        features (np.ndarray): The array of features.
        metadata (pd.DataFrame): The metadata dataframe.
        pert_col (str): The column name in the metadata dataframe representing the perturbations.
        neg_ctrl_perts (list): The list of negative control perturbations. Typically unexpressed genes.
        keys_to_drop (list, optional): The perturbation keys to be dropped from the analysis. Defaults to [].
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to 5.

    Returns:
        pd.DataFrame: The dataframe containing the query metrics.

    """
    indices = ~metadata[pert_col].isin(keys_to_drop)
    features = features[indices]
    metadata = metadata[indices]
    features_df = pd.DataFrame(features, index=metadata[pert_col])
    null_dist = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(pert_signal_consistency_metric)(np.array(features_df.loc[pert]).reshape(-1, features_df.shape[1]))
        for pert in set(metadata[pert_col]).intersection(neg_ctrl_perts)
    )
    null_dist = np.sort(null_dist)
    positive_perts = metadata[~metadata[pert_col].isin(neg_ctrl_perts)][pert_col].unique()

    def process_pert(pert):
        met, pv = pert_signal_consistency_metric(
            np.array(features_df.loc[pert]).reshape(-1, features_df.shape[1]), null_dist
        )
        return [pert, met, pv]

    query_metrics = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process_pert)(pert) for pert in positive_perts)
    return pd.DataFrame(query_metrics, columns=["pert", "avgcossim", "pval"])


def pert_signal_magnitude_metric(
    arr1: np.ndarray, arr2: np.ndarray, sorted_null: np.ndarray = np.array([])
) -> float | None | tuple[float | None, float | None]:
    """
    Calculate the perturbation signal magnitude metric, i.e., energy distance and associated p-value,
        for the two given arrays.

    Args:
        arr1 (numpy.ndarray): The feature array for the perturbation replicates.
        arr2 (numpy.ndarray): The feature array for the control replicates.
        sorted_null (numpy.ndarray, optional): Null distribution of the metric. Defaults to an empty array.
            If not empty, required to be sorted in ascending order prior to passing to this function.

    Returns:
        Union[Optional[float], tuple[Optional[float], Optional[float]]]:
        - If null is empty, returns the energy distance between arr1 and arr2 as a float.
            If the length of the input array is less than 5, returns None.
        - If null is not empty, returns a tuple containing the energy distance and p-value of the metric.
            If the length of the input array is less than 5, returns (None, None).
    """
    if len(arr1) < 5:
        return np.nan if len(sorted_null) == 0 else (np.nan, np.nan)
    edist = SamplesLoss("energy")(from_numpy(arr1), from_numpy(arr2)).item() * 2
    if len(sorted_null) == 0:
        return edist
    else:
        pval = 1 - np.searchsorted(sorted_null, edist, side="right") / len(sorted_null)
        return edist, pval


def pert_signal_magnitude_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    neg_ctrl_perts: list,
    control_key: str,
    max_controls: int = 1000,
    keys_to_drop: list = [],
    n_jobs: int = 5,
) -> pd.DataFrame:
    """
    Perform perturbation signal magnitude benchmarking, comparing the controls to the perturbations
    using the energy distance.
    Filter out perturbations specified in the `keys_to_drop` list.
    Use negative control perturbations specified in the `neg_ctrl_perts` list for the null distribution.

    Args:
        features (np.ndarray): Array of features.
        metadata (pd.DataFrame): DataFrame containing metadata.
        pert_col (str): Column name for perturbation.
        neg_ctrl_perts (list): List of negative control perturbations.
        control_key (str): Control key value.
        max_controls (int, optional): Maximum number of control perturbations to sample so energy distance
            computation runs efficiently. Defaults to 1000.
        keys_to_drop (list, optional): List of column names to drop from metadata. Should not include control_key.
            Defaults to [].
        n_jobs (int, optional): Number of parallel jobs. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame containing query metrics.

    Raises:
        ValueError: If control_key is in keys_to_drop.
    """
    if control_key in keys_to_drop:
        raise ValueError("control_key should not be in keys_to_drop.")
    indices = ~metadata[pert_col].isin(keys_to_drop)
    features = features[indices]
    metadata = metadata[indices]
    features_df = pd.DataFrame(features, index=metadata[pert_col]).sort_index()
    cf_df = features_df.loc[control_key]
    cf = np.array(cf_df.sample(min(max_controls, len(cf_df))))
    del cf_df
    null_dist = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(pert_signal_magnitude_metric)(np.array(features_df.loc[pert]).reshape(-1, features_df.shape[1]), cf)
        for pert in set(metadata[pert_col]).intersection(neg_ctrl_perts)
    )
    null_dist = np.sort(null_dist)
    positive_perts = metadata[~metadata[pert_col].isin(neg_ctrl_perts)][pert_col].unique()

    def process_pert(pert):
        met, pv = pert_signal_magnitude_metric(
            np.array(features_df.loc[pert]).reshape(-1, features_df.shape[1]), cf, null_dist
        )
        return [pert, met, pv]

    query_metrics = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process_pert)(pert) for pert in positive_perts)
    return pd.DataFrame(query_metrics, columns=["pert", "edist", "pval"])


def filter_relationships(df: pd.DataFrame):
    """
    Filters a DataFrame of relationships between entities, removing any rows with self-relationships, ie. where
        the same entity appears in both columns, and also removing any duplicate relationships (A-B and B-A).

    Args:
        df (pd.DataFrame): DataFrame containing columns 'entity1' and 'entity2', representing the entities involved in
        each relationship.

    Returns:
        pd.DataFrame: DataFrame containing columns 'entity1' and 'entity2', representing the entities involved in
        each relationship after removing any rows where the same entity appears in both columns.
    """
    df["sorted_entities"] = df.apply(lambda row: tuple(sorted([row.entity1, row.entity2])), axis=1)
    df["entity1"] = df.sorted_entities.apply(lambda x: x[0])
    df["entity2"] = df.sorted_entities.apply(lambda x: x[1])
    return df[["entity1", "entity2"]].query("entity1!=entity2").drop_duplicates()


def get_benchmark_relationships(benchmark_data_dir: str, src: str, filter=True):
    """
    Reads a CSV file containing benchmark data and returns a filtered DataFrame.

    Args:
        benchmark_data_dir (str): The directory containing the benchmark data files.
        src (str): The name of the source containing the benchmark data.
        filter (bool, optional): Whether to filter the DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark relationships.
    """
    df = pd.read_csv(Path(benchmark_data_dir).joinpath(src + ".txt"))
    return filter_relationships(df) if filter else df


def compute_recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
    recall_threshold_pairs: list,
) -> dict:
    """Compute recall at given percentage thresholds for a query distribution with respect to a null distribution.
    Each recall threshold is a pair of floats (left, right) where left and right are floats between 0 and 1.

    Args:
        null_distribution (np.ndarray): The null distribution to compare against
        query_distribution (np.ndarray): The query distribution
        recall_threshold_pairs (list) A list of pairs of floats (left, right) that represent different recall threshold
            pairs, where left and right are floats between 0 and 1.

    Returns:
        dict: A dictionary of metrics with the following keys:
            - null_distribution_size: the size of the null distribution
            - query_distribution_size: the size of the query distribution
            - recall_{left_threshold}_{right_threshold}: recall at the given percentage threshold pair(s)
    """

    metrics = {}
    metrics["null_distribution_size"] = null_distribution.shape[0]
    metrics["query_distribution_size"] = query_distribution.shape[0]

    sorted_null_distribution = np.sort(null_distribution)
    query_percentage_ranks_left = np.searchsorted(sorted_null_distribution, query_distribution, side="left") / len(
        sorted_null_distribution
    )
    query_percentage_ranks_right = np.searchsorted(sorted_null_distribution, query_distribution, side="right") / len(
        sorted_null_distribution
    )
    for threshold_pair in recall_threshold_pairs:
        left_threshold, right_threshold = np.min(threshold_pair), np.max(threshold_pair)
        metrics[f"recall_{left_threshold}_{right_threshold}"] = sum(
            (query_percentage_ranks_right <= left_threshold) | (query_percentage_ranks_left >= right_threshold)
        ) / len(query_distribution)
    return metrics


def convert_metrics_to_df(metrics: dict, source: str) -> pd.DataFrame:
    """
    Convert metrics dictionary to dataframe to be used in summary.

    Args:
        metrics (dict): metrics dictionary
        source (str): benchmark source name

    Returns:
        pd.DataFrame: a dataframe with metrics
    """
    metrics_dict_with_list = {key: [value] for key, value in metrics.items()}
    metrics_dict_with_list["source"] = [source]
    return pd.DataFrame.from_dict(metrics_dict_with_list)


def known_relationship_benchmark(
    map_data: Bunch,
    pert_col: str,
    benchmark_sources: list = cst.BENCHMARK_SOURCES,
    recall_thr_pairs: list = cst.RECALL_PERC_THRS,
    min_req_entity_cnt: int = cst.MIN_REQ_ENT_CNT,
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR,
    log_stats: bool = False,
) -> pd.DataFrame:
    """
    Perform benchmarking on aggregated map data against biological relationships.

    Args:
        map_data (Bunch): The map data containing `features` and `metadata` attributes.
        pert_col (str, optional): Column name for perturbation labels.
        benchmark_sources (list, optional): List of benchmark sources. Defaults to cst.BENCHMARK_SOURCES.
        recall_thr_pairs (list, optional): List of recall percentage threshold pairs. Defaults to cst.RECALL_PERC_THRS.
        min_req_entity_cnt (int, optional): Minimum required entity count for benchmarking.
            Defaults to cst.MIN_REQ_ENT_CNT.
        benchmark_data_dir (str, optional): Path to benchmark data directory. Defaults to cst.BENCHMARK_DATA_DIR.
        log_stats (bool, optional): Whether to print out the number of statistics used while computing the benchmarks.
            Defaults to False (i.e, no logging).

    Returns:
        pd.DataFrame: a dataframe with benchmarking results. The columns are:
            "source": benchmark source name
            "recall_{low}_{high}": recall at requested thresholds
    """

    if not len(benchmark_sources) > 0 and all([src in benchmark_data_dir for src in benchmark_sources]):
        ValueError("Invalid benchmark source(s) provided.")
    md = map_data.metadata
    features = map_data.features.set_index(md[pert_col]).rename_axis(index=None)
    del map_data
    if not len(features) == len(set(features.index)):
        ValueError("Duplicate perturbation labels in the map.")
    if not len(features) >= min_req_entity_cnt:
        ValueError("Not enough entities in the map for benchmarking.")
    if log_stats:
        print(len(features), "perturbations exist in the map.")

    metrics_lst = []
    cossim_matrix = pd.DataFrame(cosine_similarity(features, features), index=features.index, columns=features.index)
    cossim_values = cossim_matrix.values[np.triu_indices(cossim_matrix.shape[0], k=1)]
    for s in benchmark_sources:
        rels = get_benchmark_relationships(benchmark_data_dir, s)
        rels = rels[rels.entity1.isin(features.index) & rels.entity2.isin(features.index)]
        query_cossim = np.array([cossim_matrix.loc[e1, e2] for e1, e2 in rels.itertuples(index=False)])
        if log_stats:
            print(len(query_cossim), "relationships are used from the benchmark source", s)
        if len(query_cossim) > 0:
            metrics_lst.append(
                convert_metrics_to_df(metrics=compute_recall(cossim_values, query_cossim, recall_thr_pairs), source=s)
            )
    return pd.concat(metrics_lst, ignore_index=True)


def get_benchmark_clusters(
    benchmark_data_dir: str, source: str = "CORUM", min_genes: int = 1, map_genes: list = []
) -> dict:
    """
    Retrieves benchmark clusters from a file.

    Args:
        benchmark_data_dir (str): The directory where the benchmark data is located.
        source (str): The benchmark source identifier.
        min_genes (int, optional): The minimum number of genes required for a cluster to be included. Defaults to 1.

    Returns:
        dict: A dictionary containing the benchmark clusters, where the keys are cluster identifiers and the values are
            sets of genes.
    """
    result_dict = {}
    if source == "CORUM":
        file_path = Path(benchmark_data_dir).joinpath(source + "_clusters.tsv")
        with open(file_path) as file:
            for line in file:
                key, genes_str = line.strip().split("\t")
                result_dict[key] = set(genes_str.split())
    elif source == "GO":
        file_path = Path(benchmark_data_dir).joinpath("c5.go.v2023.2.Hs.symbols.gmt")
        with open(file_path) as f:
            for line in f:
                split_line = line.strip().split("\t")
                result_dict[split_line[0]] = set(split_line[2:])
    else:
        raise ValueError(f"Invalid benchmark source {source} provided.")

    result_dict_final = {}
    if len(map_genes) > 0:
        for key, genes_set in result_dict.items():
            gns = genes_set.intersection(map_genes)
            if len(gns) >= min_genes:
                result_dict_final[key] = sorted(gns)
    return result_dict_final


def cluster_benchmark(
    map_data: Bunch,
    pert_col: str,
    source: str = "CORUM",
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR,
    min_genes: int = 10,
):
    """
    Perform benchmarking of a map based on known biological cluster of perturbations.

    Args:
        map_data (Bunch): The data containing features and metadata.
        pert_col (str): The column name in the metadata used representing perturbation information.
        source (str, optional): The benchmark source. Defaults to "CORUM".
        benchmark_data_dir (str, optional): The directory containing benchmark data. Defaults to cst.BENCHMARK_DATA_DIR.
        min_genes (int, optional): The minimum number of genes required in a cluster. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmarking results, including cluster information, within-cluster
            cosine similarity mean, between-cluster cosine similarity mean, cluster size, not-cluster size,
            Kolmogorov-Smirnov statistic and p-value describing how different within-cluster cosine and between-cluster
            cosine similarity distributions are.
    """
    print(len(map_data.metadata), "genes in the map")
    benchmark_clusters = get_benchmark_clusters(
        benchmark_data_dir, source, min_genes, list(map_data.metadata[pert_col])
    )
    print(len(benchmark_clusters), "clusters are used from the benchmark source", source)
    results = []
    for k, cluster in benchmark_clusters.items():
        ind = map_data.metadata[pert_col].isin(cluster)
        cluster_data = map_data.features[ind.values]
        not_cluster_data = map_data.features[~ind.values]
        within_cossim_mat = cosine_similarity(cluster_data.values, cluster_data.values)
        within_cossim_mat_vals = within_cossim_mat[np.triu_indices(within_cossim_mat.shape[0], k=1)]
        between_cossim_mat_vals = cosine_similarity(cluster_data.values, not_cluster_data.values).flatten()

        ks_res = ks_2samp(within_cossim_mat_vals, between_cossim_mat_vals)
        results.append(
            [
                k,
                within_cossim_mat_vals.mean(),
                between_cossim_mat_vals.mean(),
                list(map_data.metadata[pert_col].loc[ind]) if not np.isnan(ks_res.pvalue) else [],
                len(cluster_data),
                len(not_cluster_data),
                ks_res.statistic,
                ks_res.pvalue,
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "cluster",
            "within_cossim_mean",
            "between_cossim_mean",
            "genes",
            "cluster_size",
            "not_cluster_size",
            "ks_stat",
            "ks_pval",
        ],
    )


def enrichment(
    genes,
    map_genes,
    source: str = "GO",
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR,
    min_genes: int = 10,
    pval_thr=0.01,
    corrected: bool = True,
):
    """
    Compute enrichment of a set of genes in a benchmark source.

    Args:
        genes (list): List of genes to compute enrichment for.
        all_genes_in_map (list): List of all genes in the map tested for enrichment.
        source (str, optional): The benchmark source. Defaults to "CORUM".
        benchmark_data_dir (str, optional): The directory containing the benchmark data.
            Defaults to cst.BENCHMARK_DATA_DIR.
        min_genes (int, optional): The minimum number of genes required in a benchmark cluster. Defaults to 3.
        pval_thr (float, optional): The p-value threshold for significance. Defaults to 0.01.
        corrected (bool, optional): Whether the p-values should be Bonferroni-corrected for multiple hypothesis testing.
            Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the clusters, p-values, and gene intersections that pass the
            significance threshold.
    """
    benchmark_clusters = get_benchmark_clusters(benchmark_data_dir, source, min_genes, map_genes)
    print(len(benchmark_clusters), "clusters are used from the benchmark source", source)
    pvals = []
    for k, cluster in benchmark_clusters.items():
        inter = set(genes).intersection(cluster)
        uni = set(genes).union(cluster)
        pval = hypergeom.sf(len(inter) - 1, len(map_genes), len(genes), len(cluster))
        pvals.append([k, pval, len(cluster), inter, len(inter) / len(uni)])
    pvals_df = pd.DataFrame(pvals, columns=["cluster", "pval", "cluster_size", "intersection", "jaccard"])
    if corrected:
        pvals_df["pval"] = pvals_df["pval"] * len(pvals_df)
        pvals_df["pval"] = pvals_df["pval"].apply(lambda x: min(x, 1))
    return pvals_df[pvals_df.pval <= pval_thr].sort_values("pval").reset_index(drop=True)


def compute_top_similars(map_data: Bunch, pert_col: str, pert1: str, pert2: str | None = None, topx: int = 10):
    """
    Compute the cosine similarity between perturbations in a map_data object and return the top similar perturbations.

    Args:
        map_data (Bunch): A map_data object containing the data and metadata.
        pert_col (str): The column name in the metadata that contains the perturbation labels.
        pert1 (str): The label of the perturbation for which to compute the cosine similarity.
        pert2 (str, optional): The label of a second perturbation to compare with pert1.
        topx (int, optional): The number of top similar perturbations to return.

    Returns:
        If pert2 is not provided or does not exist in the map_data, returns a DataFrame containing the top similar
            perturbations to pert1.
        If pert2 is provided and exists in the map_data, returns a tuple containing:
            - A DataFrame containing the top similar perturbations to pert1.
            - The rank of pert2 among the top similar perturbations to pert1.
            - The cosine similarity between pert1 and pert2.
    """
    if pert1 not in map_data.metadata[pert_col].values:
        raise ValueError(f"{pert1} does not exist in this map.")
    cosi = pd.DataFrame(
        cosine_similarity(map_data.features), index=map_data.metadata[pert_col], columns=map_data.metadata[pert_col]
    )
    pert1_rels = cosi.loc[pert1].reset_index()
    pert1_rels = pert1_rels[pert1_rels[pert_col] != pert1]
    pert1_rels.columns = ["pert", "cosine_sim"]
    pert1_rels = pert1_rels.sort_values("cosine_sim", ascending=False).reset_index(drop=True)
    if pert2 is None or (pert2 is not None and pert2 not in map_data.metadata[pert_col].values):
        if pert2 is not None:
            print(f"{pert2} does not exist in this map.")
        return pert1_rels.head(topx)
    else:
        return pert1_rels.head(topx), pert1_rels[pert1_rels["pert"] == pert2].index[0] + 1, cosi.loc[pert1, pert2]


def cosine_similarity_from_map(
    compound: str, gene: str, compound_concentration: float, map_data: pd.DataFrame, pert_col: str = "perturbation"
) -> float | None:
    """
    Returns the cosine similarity between two perturbations (compound or gene).

    Args:
        compound (str): The first perturbation id (either compound or gene).
        gene (str): The second perturbation id (either compound or gene).
        compound_concentration (float): The concentration of the compound.
        map_data (pd.DataFrame): The map_data dataframe containing both compounds and genes.
        pert_col (str, optional): The column name in the map_data dataframe representing the perturbations.
            Defaults to "perturbation".

    Returns:
        float: The cosine similarity between the two perturbations.
    """
    feature_columns = [col for col in map_data.columns if col.startswith("feature_")]

    compound_data = map_data[
        (map_data["perturbation"] == compound) & (map_data["concentration"] == compound_concentration)
    ][feature_columns]
    gene_data = map_data[map_data["perturbation"] == gene][feature_columns]

    if compound_data.empty or gene_data.empty:
        return None

    compound_values = compound_data.iloc[0].values
    gene_values = gene_data.iloc[0].values

    return compound_values.dot(gene_values) / (np.linalg.norm(compound_values) * np.linalg.norm(gene_values))


def load_truth_data(benchmark_data_dir: str) -> pd.DataFrame:
    """Load the ground truth data from a CSV file."""
    truth_data_path = Path(benchmark_data_dir) / "compound_gene_interactions.csv"
    return pd.read_csv(truth_data_path)


def compute_similarities(
    truth: pd.DataFrame,
    map_data: Bunch,
    pert_col: str,
    randomize: bool = False,
) -> pd.DataFrame:
    """Compute cosine similarities between compounds and genes."""
    treatments = truth["treatment"].unique()
    genes = truth["gene_symbol"].unique()

    compound_meta = map_data.metadata[map_data.metadata[pert_col].isin(treatments)].copy()
    gene_meta = map_data.metadata[map_data.metadata[pert_col].isin(genes)].copy()

    if compound_meta.empty or gene_meta.empty:
        raise ValueError("No matching compounds or genes found in metadata.")

    if randomize:
        rng = np.random.default_rng(cst.RANDOM_SEED)
        similarities = rng.uniform(0, 1, size=(len(compound_meta), len(gene_meta)))
    else:
        compound_features = map_data.features.loc[compound_meta.index]
        gene_features = map_data.features.loc[gene_meta.index]
        similarities = np.abs(cosine_similarity(compound_features, gene_features))

    index = pd.MultiIndex.from_arrays(
        [compound_meta[pert_col].values, compound_meta["concentration"].values],
        names=[pert_col, "concentration"],
    )

    return pd.DataFrame(similarities, index=index, columns=gene_meta[pert_col].values)


def compute_baseline_predictions(
    truth: pd.DataFrame,
    activity_threshold: float,
    inactivity_threshold: float,
    config: BenchmarkConfig,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Generate baseline predictions for evaluation."""
    rng = np.random.default_rng(config.random_seed)
    predictions = {conc: [] for conc in cst.COMPOUND_CONCENTRATIONS + ["max"]}

    if config.aggregate_by == AggregateBy.COMPOUND:
        # Baseline predictions for aggregation by compound
        all_genes = set(truth["gene_symbol"].unique())
        target_col = "gene_symbol"
        pool = all_genes
        items = truth["treatment"].unique()
    else:
        # Baseline predictions for aggregation by gene
        all_compounds = set(truth["treatment"].unique())
        target_col = "treatment"
        pool = all_compounds
        items = truth["gene_symbol"].unique()

    for item in items:
        item_data = truth[truth[target_col] == item]
        targets, labels = sample_for_item(
            item_data,
            pool,
            activity_threshold,
            inactivity_threshold,
            target_col,
            config.min_negatives,
            config.random_seed,
        )

        if len(targets) == 0:
            continue

        for conc in predictions.keys():
            scores = rng.random(len(targets))
            predictions[conc].append((scores, labels))

    return predictions


def aggregate_predictions(
    predictions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    config: BenchmarkConfig,
) -> Dict[str, Dict[str, float]]:
    """Aggregate predictions across compounds or genes."""
    results = {}
    for conc, preds in predictions.items():
        if not preds:
            results[conc] = {"average_precision": 0.0, "auc_roc": 0.5}
            continue

        if config.average_type == AverageType.MICRO:
            scores = np.concatenate([p[0] for p in preds])
            labels = np.concatenate([p[1] for p in preds])
            ap, auc = compute_metrics(scores, labels)
            results[conc] = {"average_precision": ap, "auc_roc": auc}
        else:  # MACRO
            aps = []
            aucs = []
            for scores, labels in preds:
                ap, auc = compute_metrics(scores, labels)
                aps.append(ap)
                aucs.append(auc)
            results[conc] = {"average_precision": np.mean(aps), "auc_roc": np.mean(aucs)}
    return results


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute average precision and AUC-ROC."""
    if len(scores) == 0 or not np.any(labels):
        return 0.0, 0.5

    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    tp_cumsum = np.cumsum(sorted_labels)
    precision = tp_cumsum / np.arange(1, len(sorted_labels) + 1)
    ap = np.sum(precision * sorted_labels) / sorted_labels.sum()
    auc = roc_auc_score(labels, scores)
    return ap, auc


def sample_for_item(
    item_data: pd.DataFrame,
    pool: Set[str],
    activity_threshold: float,
    inactivity_threshold: float,
    target_col: str,
    min_negatives: int = 20,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic sampling function for both compounds and genes."""
    rng = np.random.default_rng(random_seed)

    actives = item_data.loc[item_data["nM_value"] <= activity_threshold, target_col].unique()
    if len(actives) == 0:
        return np.array([]), np.array([])

    ineligibles = item_data.loc[
        (item_data["nM_value"] > activity_threshold) & (item_data["nM_value"] <= inactivity_threshold), target_col
    ].unique()

    eligibles = pool - set(actives) - set(ineligibles)
    n_negatives = max(2 * len(actives), min_negatives)

    if len(eligibles) < n_negatives:
        return np.array([]), np.array([])

    negatives = rng.choice(list(eligibles), n_negatives, replace=False)
    items = np.concatenate([actives, negatives])
    labels = np.isin(items, actives).astype(int)

    return items, labels


def process_predictions(
    data: pd.DataFrame,
    similarities: pd.DataFrame,
    config: BenchmarkConfig,
    thresholds: Tuple[float, float],
    pert_col: str = "perturbation",
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Process predictions for either compounds or genes."""
    activity_threshold, inactivity_threshold = thresholds
    predictions = {conc: [] for conc in cst.COMPOUND_CONCENTRATIONS + ["max"]}

    if config.aggregate_by == AggregateBy.COMPOUND:
        pool = set(data["gene_symbol"].unique())
        item_col, target_col = "treatment", "gene_symbol"
    else:
        pool = set(data["treatment"].unique())
        item_col, target_col = "gene_symbol", "treatment"

    for item in data[item_col].unique():
        if config.aggregate_by == AggregateBy.COMPOUND and item not in similarities.index.get_level_values(pert_col):
            continue

        item_data = data[data[item_col] == item]
        targets, labels = sample_for_item(
            item_data,
            pool,
            activity_threshold,
            inactivity_threshold,
            target_col,
            config.min_negatives,
            config.random_seed,
        )

        if len(targets) == 0:
            continue

        scores_by_conc = {}
        if config.aggregate_by == AggregateBy.COMPOUND:
            item_similarities = similarities.loc[item]
            for conc in item_similarities.index.unique():
                scores = item_similarities.loc[conc, targets].values
                if not np.all(np.isnan(scores)):
                    predictions[conc].append((scores, labels))
                    scores_by_conc[conc] = scores
        else:
            for conc in cst.COMPOUND_CONCENTRATIONS:
                try:
                    sim_conc = similarities.xs(conc, level="concentration")
                    available = sim_conc.index.intersection(targets)
                    if not available.empty:
                        scores = sim_conc.loc[available, item].values
                        labels_filtered = labels[np.isin(targets, available)]
                        if not np.all(np.isnan(scores)):
                            predictions[conc].append((scores, labels_filtered))
                            scores_by_conc[conc] = pd.Series(scores, index=available)
                except KeyError:
                    continue

        if scores_by_conc:
            if config.aggregate_by == AggregateBy.COMPOUND:
                max_scores = np.nanmax(np.vstack(list(scores_by_conc.values())), axis=0)
                if not np.all(np.isnan(max_scores)):
                    predictions["max"].append((max_scores, labels))
            else:
                available = set().union(*[scores_by_conc[conc].index for conc in scores_by_conc])
                if available:
                    scores_df = pd.DataFrame(
                        {conc: scores_by_conc[conc] for conc in scores_by_conc}, index=list(available)
                    )
                    max_scores = scores_df.max(axis=1).values
                    labels_filtered = labels[np.isin(targets, available)]
                    if not np.all(np.isnan(max_scores)):
                        predictions["max"].append((max_scores, labels_filtered))

    return predictions


def compound_gene_benchmark(
    map_data: Bunch,
    activity_threshold: float = 1000,
    inactivity_threshold: float = 10000,
    pert_col: str = "perturbation",
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR,
    truth_data: Optional[pd.DataFrame] = None,
    check_random: bool = False,
    config: Optional[BenchmarkConfig] = None,
) -> pd.DataFrame:
    """Main benchmark function."""
    config = config or BenchmarkConfig()
    truth = truth_data if truth_data is not None else load_truth_data(benchmark_data_dir)
    similarities = compute_similarities(truth, map_data, pert_col, randomize=check_random)

    thresholds = (activity_threshold, inactivity_threshold)
    predictions = process_predictions(truth, similarities, config, thresholds, pert_col)
    baseline_preds = compute_baseline_predictions(truth, activity_threshold, inactivity_threshold, config)

    results_dict = aggregate_predictions(predictions, config)
    baseline_dict = aggregate_predictions(baseline_preds, config)

    results = pd.DataFrame.from_dict(results_dict, orient="index").reset_index()
    results.rename(columns={"index": "concentration"}, inplace=True)

    baseline = pd.DataFrame.from_dict(baseline_dict, orient="index")
    results["baseline_average_precision"] = baseline["average_precision"].values
    results["baseline_auc_roc"] = baseline["auc_roc"].values

    return results




@pytest.fixture
def sample_truth_data():
    """Create sample ground truth data with known properties."""
    return pd.DataFrame({
        'treatment': ['compound1', 'compound1', 'compound2', 'compound2'],
        'gene_symbol': ['gene1', 'gene2', 'gene1', 'gene2'],
        'nM_value': [100, 5000, 15000, 500],  # Mix of active, gray zone, and inactive
    })

@pytest.fixture
def sample_map_data():
    """Create sample embedding data with known similarities."""
    features = pd.DataFrame({
        'feat1': [1.0, 0.0, 0.0, 1.0],
        'feat2': [0.0, 1.0, 1.0, 0.0]
    }, index=['compound1_id', 'compound2_id', 'gene1_id', 'gene2_id'])
    
    metadata = pd.DataFrame({
        'perturbation': ['compound1', 'compound2', 'gene1', 'gene2'],
        'concentration': ['1.0', '1.0', np.nan, np.nan],
    }, index=['compound1_id', 'compound2_id', 'gene1_id', 'gene2_id'])
    
    return Bunch(features=features, metadata=metadata)

def test_compute_similarities_basic(sample_map_data):
    """Test basic similarity computation."""
    truth = pd.DataFrame({
        'treatment': ['compound1', 'compound2'],
        'gene_symbol': ['gene1', 'gene2']
    })
    
    sims = compute_similarities(truth, sample_map_data, 'perturbation')
    
    assert isinstance(sims, pd.DataFrame)
    assert sims.shape == (2, 2)  # 2 compounds x 2 genes
    # compound1 should be perfectly similar to gene1 (same features)
    assert np.isclose(sims.loc[('compound1', '1.0'), 'gene1'], 1.0)

def test_compute_similarities_randomized(sample_map_data):
    """Test randomized similarity computation."""
    truth = pd.DataFrame({
        'treatment': ['compound1'],
        'gene_symbol': ['gene1']
    })
    
    sims = compute_similarities(truth, sample_map_data, 'perturbation', randomize=True)
    
    assert isinstance(sims, pd.DataFrame)
    assert sims.shape == (1, 1)
    assert 0 <= sims.iloc[0, 0] <= 1

def test_sample_for_item():
    """Test negative sampling logic."""
    item_data = pd.DataFrame({
        'gene_symbol': ['gene1', 'gene2', 'gene3'],
        'nM_value': [100, 5000, 15000]
    })
    pool = {'gene1', 'gene2', 'gene3', 'gene4', 'gene5'}
    
    items, labels = sample_for_item(
        item_data,
        pool,
        activity_threshold=1000,
        inactivity_threshold=10000,
        target_col='gene_symbol',
        min_negatives=2,
        random_seed=42
    )
    
    assert len(items) > 0
    assert len(labels) == len(items)
    assert sum(labels) == 1  # Only gene1 should be positive
    assert 'gene3' not in items  # Should be excluded as it's in gray zone

def test_compute_metrics():
    """Test metric computation with known values."""
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 0, 0, 1])
    
    ap, auc = compute_metrics(scores, labels)
    
    assert 0 <= ap <= 1
    assert 0 <= auc <= 1
    # With these specific values, AP should be less than 0.75
    assert ap < 0.75

def test_full_benchmark_macro_compound(sample_truth_data, sample_map_data):
    """Test full benchmark with macro averaging by compound."""
    config = BenchmarkConfig(
        average_type=AverageType.MACRO,
        aggregate_by=AggregateBy.COMPOUND,
        min_negatives=2,
        random_seed=42
    )
    
    results = compound_gene_benchmark(
        map_data=sample_map_data,
        activity_threshold=1000,
        inactivity_threshold=10000,
        truth_data=sample_truth_data,
        config=config
    )
    
    assert isinstance(results, pd.DataFrame)
    assert "concentration" in results.columns
    assert "average_precision" in results.columns
    assert "auc_roc" in results.columns
    assert "baseline_average_precision" in results.columns
    assert "baseline_auc_roc" in results.columns
    assert len(results) > 0
    assert "max" in results["concentration"].values

def test_full_benchmark_micro_gene(sample_truth_data, sample_map_data):
    """Test full benchmark with micro averaging by gene."""
    config = BenchmarkConfig(
        average_type=AverageType.MICRO,
        aggregate_by=AggregateBy.GENE,
        min_negatives=2,
        random_seed=42
    )
    
    results = compound_gene_benchmark(
        map_data=sample_map_data,
        activity_threshold=1000,
        inactivity_threshold=10000,
        truth_data=sample_truth_data,
        config=config
    )
    
    assert isinstance(results, pd.DataFrame)
    assert all(results["baseline_auc_roc"] == 0.5)  # Random baseline should have 0.5 AUC-ROC

def test_benchmark_edge_cases(sample_map_data):
    """Test benchmark behavior with edge cases."""
    # Empty truth data
    empty_truth = pd.DataFrame(columns=['treatment', 'gene_symbol', 'nM_value'])
    with pytest.raises(ValueError):
        compound_gene_benchmark(
            map_data=sample_map_data,
            truth_data=empty_truth
        )
    
    # All inactive data
    all_inactive = pd.DataFrame({
        'treatment': ['compound1'],
        'gene_symbol': ['gene1'],
        'nM_value': [20000]  # Above inactivity threshold
    })
    results = compound_gene_benchmark(
        map_data=sample_map_data,
        truth_data=all_inactive
    )
    assert len(results) > 0
    assert all(results["average_precision"] == 0.0)  # No positives should give 0 AP

def test_process_predictions(sample_truth_data, sample_map_data):
    """Test prediction processing logic."""
    config = BenchmarkConfig()
    similarities = compute_similarities(sample_truth_data, sample_map_data, 'perturbation')
    
    predictions = process_predictions(
        sample_truth_data,
        similarities,
        config,
        thresholds=(1000, 10000),
        pert_col='perturbation'
    )
    
    assert isinstance(predictions, dict)
    assert "max" in predictions
    assert all(isinstance(p, list) for p in predictions.values())
    
    # Check prediction format
    for conc, preds in predictions.items():
        if preds:
            scores, labels = preds[0]
            assert isinstance(scores, np.ndarray)
            assert isinstance(labels, np.ndarray)
            assert len(scores) == len(labels)
            assert set(labels).issubset({0, 1})

def test_config_validation():
    """Test configuration validation."""
    # Invalid average type
    with pytest.raises(ValueError):
        BenchmarkConfig(average_type="invalid")
    
    # Invalid aggregate by
    with pytest.raises(ValueError):
        BenchmarkConfig(aggregate_by="invalid")
    
    # Invalid min negatives
    with pytest.raises(ValueError):
        BenchmarkConfig(min_negatives=-1)