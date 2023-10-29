import random

import pandas as pd
from sklearn.utils import Bunch

import efaar_benchmarking.constants as cst
from efaar_benchmarking.utils import (
    compute_recall,
    convert_metrics_to_df,
    generate_null_cossims,
    generate_query_cossims,
    get_benchmark_relationships,
)


def pert_stats(
    map_data: Bunch,
    pert_pval_thr: float = cst.PERT_SIG_PVAL_THR,
):
    """
    Calculate perturbation statistics based on the provided map data.

    Args:
        map_data (Bunch): Map data containing metadata.
        pert_pval_thr (float): Perturbation significance threshold. Defaults to cst.PERT_SIG_PVAL_THR.

    Returns:
        dict: Dictionary containing perturbation statistics:
            - "all_pert_count": Total count of perturbations.
            - "pp_pert_count": Count of perturbations that meet the significance threshold.
            - "pp_pert_percent": Percentage of perturbations that meet the significance threshold.
    """

    md = map_data.metadata
    idx = [True] * len(md)
    pidx = md[cst.PERT_SIG_PVAL_COL] <= pert_pval_thr
    return {
        "all_pert_count": sum(idx),
        "pp_pert_count": sum(idx & pidx),
        "pp_pert_percent": sum(idx & pidx) / sum(idx),
    }


def benchmark(
    map_data: Bunch,
    benchmark_sources: list = cst.BENCHMARK_SOURCES,
    pert_label_col: str = cst.PERT_LABEL_COL,
    recall_thr_pairs: list = cst.RECALL_PERC_THRS,
    filter_on_pert_prints: bool = False,
    pert_pval_thr: float = cst.PERT_SIG_PVAL_THR,
    n_null_samples: int = cst.N_NULL_SAMPLES,
    random_seed: int = cst.RANDOM_SEED,
    n_iterations: int = cst.RANDOM_COUNT,
    min_req_entity_cnt: int = cst.MIN_REQ_ENT_CNT,
    benchmark_data_dir: str = cst.BENCHMARK_DATA_DIR
) -> pd.DataFrame:
    """Perform benchmarking on map data.

    Args:
        map_data (Bunch): The map data containing `features` and `metadata` attributes.
        benchmark_sources (list, optional): List of benchmark sources. Defaults to cst.BENCHMARK_SOURCES.
        pert_label_col (str, optional): Column name for perturbation labels. Defaults to cst.PERT_LABEL_COL.
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
        AssertionError("Invalid benchmark source(s) provided.")
    md = map_data.metadata
    idx = (md[cst.PERT_SIG_PVAL_COL] <= pert_pval_thr) if filter_on_pert_prints else [True] * len(md)
    features = map_data.features[idx].set_index(md[idx][pert_label_col]).rename_axis(index=None)
    del map_data
    if not len(features) == len(set(features.index)):
        AssertionError("Duplicate perturbation labels in the map.")
    if not len(features) >= min_req_entity_cnt:
        AssertionError("Not enough entities in the map for benchmarking.")
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
            query_cossim = generate_query_cossims(features, get_benchmark_relationships(benchmark_data_dir, s))
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
