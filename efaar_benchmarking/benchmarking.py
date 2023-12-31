import random
from pathlib import Path
from time import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from geomloss import SamplesLoss
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import Bunch
from torch import from_numpy

import efaar_benchmarking.constants as cst


def univariate_consistency_metric(
    arr: np.ndarray, null: np.ndarray = np.array([])
) -> Union[Optional[float], tuple[Optional[float], Optional[float]]]:
    """
    Calculate the univariate consistency metric, i.e. average cosine angle and associated p-value, for a given array.

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


def univariate_consistency_benchmark(
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
                delayed(univariate_consistency_metric)(
                    np.random.default_rng(random_seed + r).choice(features, c, False)
                )
                for r in range(n_samples)
            )
            for c in unique_cardinalities
        }
        query_metrics = features_df.groupby(features_df.index, observed=True).apply(
            lambda x: univariate_consistency_metric(x.values, null[len(x)])[1]  # type: ignore[index]
        )
    else:
        cardinalities_df = metadata.groupby(by=[pert_col, batch_col], observed=True).size().reset_index(name="count")
        df_perts = (
            cardinalities_df.groupby(by=pert_col, observed=True)[[batch_col, "count"]]
            .apply(lambda x: list(map(tuple, x.values)))
            .reset_index()
        )
        nulls_b_cnt = {}
        null_pert = {}
        features_df_batch = pd.DataFrame(features, index=metadata[batch_col])
        np.random.seed(random_seed)
        for pert, bscnts in df_perts.itertuples(index=False):
            for b, c in bscnts:
                if (b, c) not in nulls_b_cnt:
                    # reshape call in the line below is for the outlier case of when a batch has only 1 sample.
                    bfeat = np.array(features_df_batch.loc[b]).reshape(-1, features_df_batch.shape[1])
                    nulls_b_cnt[(b, c)] = bfeat[np.random.randint(0, bfeat.shape[0], (n_samples, c))]
            result_array = np.concatenate([nulls_b_cnt[(b, c)] for b, c in bscnts], axis=1)
            t = time()
            null_pert[pert] = Parallel(n_jobs=n_jobs)(delayed(univariate_consistency_metric)(r) for r in result_array)
            print(f"{pert} has {result_array.shape[1]} samples and took {round(time()-t, 2)} seconds.")
        query_metrics = features_df.groupby(features_df.index, observed=True).apply(
            lambda x: univariate_consistency_metric(x.values, null_pert[x.name])[1]  # type: ignore[index]
        )
    query_metrics.name = "avg_cossim_pval"
    return query_metrics.reset_index()


def univariate_distance_metric(
    arr1: np.ndarray, arr2: np.ndarray, null: np.ndarray = np.array([])
) -> Union[Optional[float], tuple[Optional[float], Optional[float]]]:
    """
    Calculate the univariate distance metric, i.e. energy distance and associated p-value, for the two given arrays.

    Args:
        gf (numpy.ndarray): The feature array for the perturbation replicates.
        cf (numpy.ndarray): The feature array for the control replicates.
        null (numpy.ndarray, optional): Null distribution of the metric. Defaults to an empty array.

    Returns:
        Union[Optional[float], tuple[Optional[float], Optional[float]]]:
        - If null is empty, returns the energy distance between arr1 and arr2 as a float.
            If the length of the input array is less than 10, returns None.
        - If null is not empty, returns a tuple containing the energy distance and p-value of the metric.
            If the length of the input array is less than 10, returns (None, None).
    """
    if len(arr1) < 10:
        return None if len(null) == 0 else None, None
    edist = SamplesLoss("energy")(from_numpy(arr1), from_numpy(arr2)).item() * 2
    if len(null) == 0:
        return edist
    else:
        sorted_null = np.sort(null)
        pval = 1 - np.searchsorted(sorted_null, edist, side="right") / len(sorted_null)
        return edist, pval


def univariate_distance_metric_null(rng, combined_array, len_arr1):
    """
    Calculate the univariate distance metric for two arrays in a combined array, given a random number generator.

    Parameters:
    - rng: The random number generator.
    - combined_array: The combined input array.
    - len_arr1: The length of the first part of the input array.

    Returns:
    - The univariate distance metric between the first and second part of the combined array.
    """
    indices = rng.choice(len(combined_array), len(combined_array), replace=False)
    return univariate_distance_metric(combined_array[indices[:len_arr1], :], combined_array[indices[len_arr1:], :])


def univariate_distance_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: Optional[str] = None,
    n_samples: int = cst.N_NULL_SAMPLES,
    random_seed: int = cst.RANDOM_SEED,
    n_jobs: int = 5,
) -> pd.DataFrame:
    if batch_col is None:
        print("TODO")
    else:
        cardinalities_df = metadata.groupby(by=[pert_col, batch_col], observed=True).size().reset_index(name="count")
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
                    controls_b_c[b, c] = np.array(features_df.loc[(control_key, b)])
            cf = np.concatenate([controls_b_c[(b, c)] for b, c in bscnts], axis=0)
            af = np.concatenate((cf, gf))
            t = time()
            null_dist = Parallel(n_jobs=n_jobs)(
                delayed(univariate_distance_metric_null)(np.random.default_rng(random_seed + r), af, len(gf))
                for r in range(n_samples)
            )
            print(f"{pert} has {len(gf)} samples and took {round(time()-t, 2)} seconds.")
            query_metrics.append([pert, univariate_distance_metric(gf, cf, null_dist)[1]])  # type: ignore[index]
        return pd.DataFrame(query_metrics, columns=["pert", "avg_edist_pval"])


def compute_process_cosine_sim(
    entity1_feats: pd.DataFrame,
    entity2_feats: pd.DataFrame,
    filter_to_pairs: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of entities' features.

    Args:
        entity1_feats (pd.DataFrame): Features of the first set of entities.
        entity2_feats (pd.DataFrame): Features of the second set of entities.
        filter_to_pairs (Optional[pd.DataFrame], optional): DataFrame to filter the pairs to. Defaults to None.

    Returns:
        np.ndarray: A NumPy array containing the cosine similarity values between the pairs of entities.
    """

    cosi = pd.DataFrame(
        cosine_similarity(entity1_feats, entity2_feats), index=entity1_feats.index, columns=entity2_feats.index
    )
    # convert pairwise cosine similarity matrix to a data frame of triples so that filtering and grouping is easy
    cosi = cosi.stack()[np.ones(cosi.size).astype("bool")].reset_index()
    cosi.columns = ["entity1", "entity2", "cosine_sim"]  # type: ignore
    if filter_to_pairs is not None:
        cosi = cosi.merge(filter_to_pairs, how="right", on=["entity1", "entity2"])
    cosi = cosi[cosi.entity1 != cosi.entity2]  # remove self cosine similarities
    return cosi.cosine_sim.values  # type: ignore


def generate_null_cossims(
    feats: pd.DataFrame,
    n_sample_entities: int,
    rseed_entity1: int,
    rseed_entity2: int,
) -> np.ndarray:
    """Generate null cosine similarity values between randomly sampled subsets of entities.

    Args:
        feats (pd.DataFrame): Features of the first set of entities.
        n_entities (int): Number of entities to sample for null.
        rseed_entity1 (int): Random seed for sampling subset from entity1_feats.
        rseed_entity2 (int): Random seed for sampling subset from entity2_feats.

    Returns:
        np.ndarray: A NumPy array containing the null cosine similarity values between the randomly sampled subsets
            of entities.
    """

    np.random.seed(rseed_entity1)
    entity1_feats = feats.loc[np.random.choice(list(feats.index.unique()), n_sample_entities)]
    np.random.seed(rseed_entity2)
    entity2_feats = feats.loc[np.random.choice(list(feats.index.unique()), n_sample_entities)]
    return compute_process_cosine_sim(entity1_feats, entity2_feats)


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


def generate_query_cossims(
    feats: pd.DataFrame,
    gt_source_df: pd.DataFrame,
    min_req_entity_cnt: int = cst.MIN_REQ_ENT_CNT,
) -> np.ndarray:
    """Generate query-specific cosine similarity values between subsets of entities' features.

    Args:
        feats (pd.DataFrame): Features of the first set of entities.
        gt_source_df (pd.DataFrame): DataFrame containing ground truth annotation sources.
        min_req_entity_cnt (int, optional): Minimum required entity count for benchmarking.
            Defaults to cst.MIN_REQ_ENT_CNT.

    Returns:
        np.ndarray: A NumPy array containing the query-specific cosine similarity values, or None
            if there are not enough entities for benchmarking.
    """

    gt_source_df = gt_source_df[gt_source_df.entity1.isin(feats.index) & gt_source_df.entity2.isin(feats.index)]
    entity1_feats = feats.loc[list(set(gt_source_df.entity1))]
    entity2_feats = feats.loc[list(set(gt_source_df.entity2))]
    if len(set(entity1_feats.index)) >= min_req_entity_cnt and len(set(entity2_feats.index)) >= min_req_entity_cnt:
        return compute_process_cosine_sim(entity1_feats, entity2_feats, gt_source_df)
    else:
        return np.empty(shape=(0, 0))


def compute_recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
    recall_threshold_pairs: list,
    right_sided: bool = False,
) -> dict:
    """Compute recall at given percentage thresholds for a query distribution with respect to a null distribution.
    Each recall threshold is a pair of floats (left, right) where left and right are floats between 0 and 1.

    Args:
        null_distribution (np.ndarray): The null distribution to compare against
        query_distribution (np.ndarray): The query distribution
        recall_threshold_pairs (list) A list of pairs of floats (left, right) that represent different recall threshold
            pairs, where left and right are floats between 0 and 1.
        right_sided (bool, optional): Whether to consider only right tail of the distribution or both tails when
            computing recall Defaults to False (i.e, both tails).

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
        if right_sided:
            metrics[f"recall_{left_threshold}_{right_threshold}"] = sum(
                query_percentage_ranks_left >= right_threshold
            ) / len(query_distribution)
        else:
            metrics[f"recall_{left_threshold}_{right_threshold}"] = sum(
                (query_percentage_ranks_right <= left_threshold) | (query_percentage_ranks_left >= right_threshold)
            ) / len(query_distribution)
    return metrics


def convert_metrics_to_df(
    metrics: dict,
    source: str,
    random_seed_str: str,
    filter_on_pert_prints: bool,
) -> pd.DataFrame:
    """
    Convert metrics dictionary to dataframe to be used in summary.

    Args:
        metrics (dict): metrics dictionary
        source (str): benchmark source name
        random_seed_str (str): random seed string from random seeds 1 and 2
        filter_on_pert_prints (bool): whether metrics were computed after filtering on perturbation prints or not

    Returns:
        pd.DataFrame: a dataframe with metrics
    """
    metrics_dict_with_list = {key: [value] for key, value in metrics.items()}
    metrics_dict_with_list["source"] = [source]
    metrics_dict_with_list["random_seed"] = [random_seed_str]
    metrics_dict_with_list["filter_on_pert_prints"] = [filter_on_pert_prints]
    return pd.DataFrame.from_dict(metrics_dict_with_list)


def multivariate_benchmark(
    map_data: Bunch,
    pert_col: str,
    benchmark_sources: list = cst.BENCHMARK_SOURCES,
    recall_thr_pairs: list = cst.RECALL_PERC_THRS,
    filter_on_pert_prints: bool = False,
    pert_pval_thr: float = cst.PERT_SIG_PVAL_THR,
    n_null_samples: int = cst.N_NULL_SAMPLES,
    random_seed: int = cst.RANDOM_SEED,
    n_iterations: int = cst.RANDOM_COUNT,
    min_req_entity_cnt: int = cst.MIN_REQ_ENT_CNT,
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR,
    right_sided: bool = False,
    log_stats: bool = False,
) -> pd.DataFrame:
    """Perform benchmarking on map data.

    Args:
        map_data (Bunch): The map data containing `features` and `metadata` attributes.
        pert_col (str, optional): Column name for perturbation labels.
        benchmark_sources (list, optional): List of benchmark sources. Defaults to cst.BENCHMARK_SOURCES.
        recall_thr_pairs (list, optional): List of recall percentage threshold pairs. Defaults to cst.RECALL_PERC_THRS.
        filter_on_pert_prints (bool, optional): Flag to filter map data based on perturbation prints. Defaults to False.
        pert_pval_thr (float, optional): pvalue threshold for perturbation filtering. Defaults to cst.PERT_SIG_PVAL_THR.
        n_null_samples (int, optional): Number of null samples to generate. Defaults to cst.N_NULL_SAMPLES.
        random_seed (int, optional): Random seed to use for generating null samples. Defaults to cst.RANDOM_SEED.
        n_iterations (int, optional): Number of random seed pairs to use. Defaults to cst.RANDOM_COUNT.
        min_req_entity_cnt (int, optional): Minimum required entity count for benchmarking.
            Defaults to cst.MIN_REQ_ENT_CNT.
        benchmark_data_dir (str, optional): Path to benchmark data directory. Defaults to cst.BENCHMARK_DATA_DIR.
        right_sided (bool, optional): Whether to consider only right tail of the distribution or both tails.
            Defaults to False (i.e, both tails).
        log_stats (bool, optional): Whether to print out the number of statistics used while computing the benchmarks.
            Defaults to False (i.e, no logging).

    Returns:
        pd.DataFrame: a dataframe with benchmarking results. The columns are:
            "source": benchmark source name
            "random_seed": random seed string from random seeds 1 and 2
            "recall_{low}_{high}": recall at requested thresholds
    """

    if not len(benchmark_sources) > 0 and all([src in benchmark_data_dir for src in benchmark_sources]):
        ValueError("Invalid benchmark source(s) provided.")
    md = map_data.metadata
    idx = (md[cst.PERT_SIG_PVAL_COL] <= pert_pval_thr) if filter_on_pert_prints else [True] * len(md)
    features = map_data.features[idx].set_index(md[idx][pert_col]).rename_axis(index=None)
    del map_data
    if not len(features) == len(set(features.index)):
        ValueError("Duplicate perturbation labels in the map.")
    if not len(features) >= min_req_entity_cnt:
        ValueError("Not enough entities in the map for benchmarking.")
    if log_stats:
        print(len(features), "perturbations exist in the map.")

    metrics_lst = []
    random.seed(random_seed)
    random_seed_pairs = [
        (random.randint(0, 2**31 - 1), random.randint(0, 2**31 - 1)) for _ in range(n_iterations)  # nosec
    ]  # numpy requires seeds to be between 0 and 2 ** 32 - 1
    for rs1, rs2 in random_seed_pairs:
        random_seed_str = f"{rs1}_{rs2}"
        null_cossim = generate_null_cossims(features, n_null_samples, rs1, rs2)
        for s in benchmark_sources:
            rels = get_benchmark_relationships(benchmark_data_dir, s)
            query_cossim = generate_query_cossims(features, rels)
            if log_stats:
                print(len(query_cossim), "relationships are used from the benchmark source", s)
            if len(query_cossim) > 0:
                metrics_lst.append(
                    convert_metrics_to_df(
                        metrics=compute_recall(null_cossim, query_cossim, recall_thr_pairs, right_sided),
                        source=s,
                        random_seed_str=random_seed_str,
                        filter_on_pert_prints=filter_on_pert_prints,
                    )
                )
    return pd.concat(metrics_lst, ignore_index=True)
