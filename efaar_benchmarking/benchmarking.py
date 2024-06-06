from pathlib import Path
from time import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from geomloss import SamplesLoss
from joblib import Parallel, delayed
from scipy.stats import hypergeom, ks_2samp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import Bunch
from torch import from_numpy

import efaar_benchmarking.constants as cst


def pert_signal_consistency_metric(
    arr: np.ndarray, null: np.ndarray = np.array([])
) -> Union[Optional[float], tuple[Optional[float], Optional[float]]]:
    """
    Calculate the perturbation signal consistency metric, i.e., average cosine angle and associated p-value,
        for a given array.

    Args:
        arr (numpy.ndarray): The input array.
        null (numpy.ndarray, optional): Null distribution of the metric. Defaults to an empty array.

    Returns:
        Union[Optional[float], tuple[Optional[float], Optional[float]]]:
        - If null is empty, returns the average angle as a float. If the length of the input array is less than 2,
            returns None.
        - If null is not empty, returns a tuple containing the average angle and p-value of the metric. If the length
            of the input array is less than 2, returns (None, None).
    """
    if len(arr) < 2:
        return None if len(null) == 0 else None, None
    cosine_sim = np.clip(cosine_similarity(arr), -1, 1)  # to avoid floating point precision errors
    avg_angle = np.arccos(cosine_sim[np.tril_indices(cosine_sim.shape[0], k=-1)]).mean()
    if len(null) == 0:
        return avg_angle
    else:
        sorted_null = np.sort(null)
        pval = np.searchsorted(sorted_null, avg_angle) / len(sorted_null)
        return avg_angle, pval


def pert_signal_consistency_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    batch_col: Optional[str] = None,
    keys_to_drop: list = [],
    n_samples: int = cst.N_NULL_SAMPLES,
    random_seed: int = cst.RANDOM_SEED,
    n_jobs: int = 5,
) -> pd.DataFrame:
    """
    Perform perturbation consistency benchmarking on the given features and metadata.

    Args:
        features (np.ndarray): The array of features.
        metadata (pd.DataFrame): The metadata dataframe.
        pert_col (str): The column name in the metadata dataframe representing the perturbations.
        batch_col (str): The column name in the metadata dataframe representing the batches.
        keys_to_drop (list): The perturbation keys to be dropped from the analysis.
        n_samples (int, optional): The number of samples to generate for null distribution.
            Defaults to cst.N_NULL_SAMPLES.
        random_seed (int, optional): The random seed to use for generating null distribution.
            Defaults to cst.RANDOM_SEED.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to 5.

    Returns:
        pd.DataFrame: The dataframe containing the query metrics.
    """
    indices = ~metadata[pert_col].isin(keys_to_drop)
    features = features[indices]
    metadata = metadata[indices]
    features_df = pd.DataFrame(features, index=metadata[pert_col])
    if batch_col is None:
        unique_cardinalities = metadata.groupby(pert_col, observed=True).count().iloc[:, 0].unique()
        null = {
            c: Parallel(n_jobs=n_jobs)(
                delayed(pert_signal_consistency_metric)(
                    np.random.default_rng(random_seed + r).choice(features, c, False)
                )
                for r in range(n_samples)
            )
            for c in unique_cardinalities
        }
        query_metrics = (
            features_df.groupby(features_df.index, observed=True)
            .apply(lambda x: pert_signal_consistency_metric(x.values, null[len(x)])[1])  # type: ignore[index]
            .reset_index()
        )
        query_metrics.columns = ["pert", "pval"]
        return query_metrics
    else:
        cardinalities_df = metadata.groupby(by=[pert_col, batch_col], observed=True).size().reset_index(name="count")
        df_perts = (
            cardinalities_df.groupby(by=pert_col, observed=True)[[batch_col, "count"]]
            .apply(lambda x: list(map(tuple, x.values)))
            .reset_index()
        )
        nulls_b_cnt = {}
        features_df_batch = pd.DataFrame(features, index=metadata[batch_col])
        np.random.seed(random_seed)
        query_metrics = []
        for pert, bscnts in df_perts.itertuples(index=False):
            for b, c in bscnts:
                if (b, c) not in nulls_b_cnt:
                    # reshape call in the line below is for the outlier case of when a batch has only 1 sample.
                    bfeat = np.array(features_df_batch.loc[b]).reshape(-1, features_df_batch.shape[1])
                    nulls_b_cnt[(b, c)] = bfeat[np.random.randint(0, bfeat.shape[0], (n_samples, c))]
            result_array = np.concatenate([nulls_b_cnt[(b, c)] for b, c in bscnts], axis=1)
            t = time()
            null_dist = Parallel(n_jobs=n_jobs)(delayed(pert_signal_consistency_metric)(r) for r in result_array)
            met, pv = pert_signal_consistency_metric(  # type: ignore[misc]
                np.array(features_df.loc[pert]).reshape(-1, features_df.shape[1]), null_dist
            )
            query_metrics.append([pert, met, pv])
            print(f"{pert} has {result_array.shape[1]} samples and took {round(time()-t, 2)} seconds: {met} {pv}")
        return pd.DataFrame(query_metrics, columns=["pert", "angle", "pval"])


def pert_signal_distance_metric(
    arr1: np.ndarray, arr2: np.ndarray, null: np.ndarray = np.array([])
) -> Union[Optional[float], tuple[Optional[float], Optional[float]]]:
    """
    Calculate the perturbation signal distance metric, i.e., energy distance and associated p-value,
        for the two given arrays.

    Args:
        gf (numpy.ndarray): The feature array for the perturbation replicates.
        cf (numpy.ndarray): The feature array for the control replicates.
        null (numpy.ndarray, optional): Null distribution of the metric. Defaults to an empty array.

    Returns:
        Union[Optional[float], tuple[Optional[float], Optional[float]]]:
        - If null is empty, returns the energy distance between arr1 and arr2 as a float.
            If the length of the input array is less than 5, returns None.
        - If null is not empty, returns a tuple containing the energy distance and p-value of the metric.
            If the length of the input array is less than 5, returns (None, None).
    """
    if len(arr1) < 5:
        return None if len(null) == 0 else None, None
    edist = SamplesLoss("energy")(from_numpy(arr1), from_numpy(arr2)).item() * 2
    if len(null) == 0:
        return edist
    else:
        sorted_null = np.sort(null)
        pval = 1 - np.searchsorted(sorted_null, edist, side="right") / len(sorted_null)
        return edist, pval


def pert_signal_distance_metric_null(
    rng: np.random.Generator, combined_array: np.ndarray, len_arr1: int
) -> tuple[Optional[float], Optional[float]]:
    """
    Calculate the perturbation signal distance metric for two arrays in a combined array,
        given a random number generator.

    Args:
        rng (np.random.Generator): The random number generator.
        combined_array (np.ndarray): The combined input array.
        len_arr1 (int): The length of the first part of the input array.

    Returns:
        tuple[Optional[float], Optional[float]]:
    """
    if len(combined_array) <= len_arr1:
        raise ValueError(f"len(combined_array) needs to be larger than {len_arr1}")
    indices = rng.choice(len(combined_array), len(combined_array), replace=False)
    return pert_signal_distance_metric(
        combined_array[indices[:len_arr1], :], combined_array[indices[len_arr1:], :]
    )  # type: ignore[return-value]


def pert_signal_distance_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col,
    keys_to_drop: list = [],
    n_samples: int = cst.N_NULL_SAMPLES,
    random_seed: int = cst.RANDOM_SEED,
    n_jobs: int = 5,
) -> pd.DataFrame:
    """
    Perform perturbation signal distance benchmarking, comparing the controls to the perturbations
        using the energy distance.

    Args:
        features (np.ndarray): Array of features.
        metadata (pd.DataFrame): DataFrame containing metadata.
        pert_col (str): Column name for perturbation.
        control_key (str): Control key value.
        keys_to_drop (list, optional): List of column names to drop from metadata. Should not include control_key.
            Defaults to [].
        batch_col (str): Column name for batch.
        n_samples (int, optional): Number of null samples. Defaults to cst.N_NULL_SAMPLES.
        random_seed (int, optional): Random seed. Defaults to cst.RANDOM_SEED.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame containing query metrics.
    """
    if batch_col is None:
        raise ValueError("batch_col=None option is yet to be implemented. Please provide a batch column.")
    if control_key in keys_to_drop:
        raise ValueError("control_key should not be in keys_to_drop.")
    indices = ~metadata[pert_col].isin(keys_to_drop)
    features = features[indices]
    metadata = metadata[indices]
    rng = np.random.default_rng(random_seed)
    cardinalities_df = (
        metadata[metadata[pert_col] != control_key]
        .groupby(by=[pert_col, batch_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    df_perts = (
        cardinalities_df.groupby(by=pert_col, observed=True)[[batch_col, "count"]]
        .apply(lambda x: list(map(tuple, x.values)))
        .reset_index()
    )
    controls_b_c = {}
    features_df = pd.DataFrame(features, index=[metadata[pert_col], metadata[batch_col]]).sort_index()
    query_metrics = []
    for pert, bscnts in df_perts.itertuples(index=False):
        gf = np.array(features_df.loc[pert])
        for b, c in bscnts:
            if (b, c) not in controls_b_c:
                controls_b_c[b, c] = rng.choice(np.array(features_df.loc[(control_key, b)]), c, replace=False)
        cf = np.concatenate([controls_b_c[(b, c)] for b, c in bscnts], axis=0)
        af = np.concatenate((cf, gf))
        t = time()
        null_dist = Parallel(n_jobs=n_jobs)(
            delayed(pert_signal_distance_metric_null)(np.random.default_rng(random_seed + r), af, len(gf))
            for r in range(n_samples)
        )
        met, pv = pert_signal_distance_metric(gf, cf, null_dist)  # type: ignore[misc]
        query_metrics.append([pert, met, pv])
        print(f"{pert} has {len(gf)} samples and took {round(time()-t, 2)} seconds: {met} {pv}")
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


def get_benchmark_clusters(benchmark_data_dir: str, source: str = "CORUM", min_genes: int = 1):
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
                genes_set = set(genes_str.split())
                if len(genes_set) >= min_genes:
                    result_dict[key] = genes_set
    else:
        raise ValueError(f"Invalid benchmark source {source} provided.")
    return result_dict


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
    benchmark_clusters = get_benchmark_clusters(benchmark_data_dir, source, min_genes)
    print(len(benchmark_clusters), "clusters are used from the benchmark source", source)
    results = []
    for k, cluster in benchmark_clusters.items():
        cluster_data = map_data.features[map_data.metadata[pert_col].isin(cluster)]
        if len(cluster_data) < min_genes:
            continue
        not_cluster_data = map_data.features[~map_data.metadata[pert_col].isin(cluster)]

        within_cossim_mat = cosine_similarity(cluster_data.values, cluster_data.values)
        within_cossim_mat_vals = within_cossim_mat[np.triu_indices(within_cossim_mat.shape[0], k=1)]
        between_cossim_mat_vals = cosine_similarity(cluster_data.values, not_cluster_data.values).flatten()

        ks_res = ks_2samp(within_cossim_mat_vals, between_cossim_mat_vals)
        results.append(
            [
                k,
                within_cossim_mat_vals.mean(),
                between_cossim_mat_vals.mean(),
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
            "cluster_size",
            "not_cluster_size",
            "ks_stat",
            "ks_pval",
        ],
    )


def enrichment(
    genes,
    source: str = "CORUM",
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR,
    min_genes: int = 3,
    pval_thr=0.01,
    bg_gene_cnt=20000,
):
    """
    Compute enrichment of a set of genes in a benchmark source.

    Args:
        genes (list): List of genes to compute enrichment for.
        source (str, optional): The benchmark source. Defaults to "CORUM".
        benchmark_data_dir (str, optional): The directory containing the benchmark data.
            Defaults to cst.BENCHMARK_DATA_DIR.
        min_genes (int, optional): The minimum number of genes required in a benchmark cluster. Defaults to 3.
        pval_thr (float, optional): The p-value threshold for significance. Defaults to 0.01.
        bg_gene_cnt (int, optional): The total number of background genes. Defaults to 20000.

    Returns:
        pandas.DataFrame: A DataFrame containing the clusters, p-values, and gene intersections that pass the
            significance threshold.
    """
    benchmark_clusters = get_benchmark_clusters(benchmark_data_dir, source, min_genes)
    print(len(benchmark_clusters), "clusters are used from the benchmark source", source)
    pvals = []
    for k, cluster in benchmark_clusters.items():
        inter = set(genes).intersection(cluster)
        pval = hypergeom.sf(len(inter) - 1, bg_gene_cnt, len(genes), len(cluster))
        pvals.append([k, pval, inter, len(cluster)])
    pvals = pd.DataFrame(pvals, columns=["cluster", "pval", "intersection", "cluster size"])
    return pvals[pvals.pval <= pval_thr].sort_values("pval").reset_index(drop=True)  # type: ignore


def compute_top_similars(map_data: Bunch, pert_col: str, pert1: str, pert2: Optional[str] = None, topx: int = 10):
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
