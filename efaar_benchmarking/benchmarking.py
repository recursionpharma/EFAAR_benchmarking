import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import Bunch

import efaar_benchmarking.constants as cst
from efaar_benchmarking.utils import (
    compute_recall,
    convert_metrics_to_df,
    generate_null_cossims,
    generate_query_cossims,
    get_benchmark_relationships,
)


def univariate_consistency_metric(arr: np.ndarray, null: np.ndarray = None) -> tuple[float, float]:
    """
    Calculate the univariate consistency metric, i.e. average cosine angle and associated p-value, for a given array.

    Args:
        arr (numpy.ndarray): The input array.
        null (numpy.ndarray, optional): Null distribution of the metric. Default is None.

    Returns:
        tuple: A tuple containing the average angle (avg_angle) and p-value (pval) of the metric.
           If the length of the input array is less than 3, returns (None, None).
           If null is None, returns (avg_angle, None).
    """
    if len(arr) < 3:
        return np.nan, np.nan
    cosine_sim = cosine_similarity(arr)
    avg_angle = np.arccos(cosine_sim[np.tril_indices(cosine_sim.shape[0], k=-1)]).mean()
    if null is None:
        return avg_angle, np.nan
    else:
        sorted_null = np.sort(null)
        pval = np.searchsorted(sorted_null, avg_angle) / len(sorted_null)
        return avg_angle, pval


def univariate_consistency_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    keys_to_drop: str,
    n_samples: int = 5000,
    random_seed: int = cst.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Perform univariate consistency benchmarking on the given features and metadata.

    Args:
        features (np.ndarray): The array of features.
        metadata (pd.DataFrame): The metadata dataframe.
        pert_col (str): The column name in the metadata dataframe representing the perturbations.
        keys_to_drop (str): The perturbation keys to be dropped from the analysis.
        n_samples (int, optional): The number of samples to generate for null distribution. Defaults to 5000.

    Returns:
        pd.DataFrame: The dataframe containing the query metrics.
    """
    indices = ~metadata[pert_col].isin(keys_to_drop)
    features = features[indices]
    metadata = metadata[indices]

    unique_cardinalities = metadata.groupby(pert_col).count().iloc[:, 0].unique()
    null = {
        x: [
            univariate_consistency_metric(np.random.default_rng(seed=random_seed).choice(features, x, False))[0]
            for i in range(n_samples)
        ]
        for x in unique_cardinalities
    }

    features_df = pd.DataFrame(features, index=metadata[pert_col])
    query_metrics = features_df.groupby(features_df.index).apply(
        lambda x: univariate_consistency_metric(x.values, null[len(x)])[1]
    )
    query_metrics.name = "avg_cossim_pval"
    query_metrics = query_metrics.reset_index()

    return query_metrics


def benchmark(
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
            print(len(rels), "relationships exist in the benchmark source.")
            query_cossim = generate_query_cossims(features, rels)
            if len(query_cossim) > 0:
                metrics_lst.append(
                    convert_metrics_to_df(
                        metrics=compute_recall(null_cossim, query_cossim, recall_thr_pairs),
                        source=s,
                        random_seed_str=random_seed_str,
                        filter_on_pert_prints=filter_on_pert_prints,
                    )
                )
    return pd.concat(metrics_lst, ignore_index=True)
