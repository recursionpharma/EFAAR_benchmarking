from efaar_benchmarking.utils import compute_pairwise_metrics
import efaar_benchmarking.constants as cst
from sklearn.utils import Bunch
import pandas as pd
import numpy as np
from collections import defaultdict


def pert_stats(
    map_data: Bunch,
    pert_sig_thr: float = cst.PERT_SIG_PVAL_THR,
):
    map_data_orig = load_pheno_data(map_data) if type(map_data) == str else map_data
    md = map_data_orig.metadata
    gidx = md[cst.WELL_TYPE_COL] == cst.WELL_TYPE
    pidx = md[cst.PERT_SIG_PVAL_COL] <= pert_sig_thr
    return {
        "all_gene_count": sum(gidx),
        "pp_gene_count": sum(gidx & pidx),
        "pp_gene_percent": sum(gidx & pidx) / sum(gidx),
    }


def benchmark(
    map_data: Bunch,
    pert_label_col: str = cst.PERT_LABEL_COL,
    benchmark_sources: list = cst.BENCHMARK_SOURCES,
    filter_on_pert_type=False,
    filter_on_well_type=False,
    filter_on_pert_prints=False,
) -> dict:
    md = map_data.metadata
    idx = [True] * len(md)
    if filter_on_pert_type:
        idx = idx & (md[cst.PERT_TYPE_COL] == cst.PERT_TYPE)
    if filter_on_well_type:
        idx = idx & (md[cst.WELL_TYPE_COL] == cst.WELL_TYPE)
    if filter_on_pert_prints:
        pval_thresh = cst.PERT_SIG_PVAL_THR if filter_on_pert_prints else 1
        idx = idx & (md[cst.PERT_SIG_PVAL_COL] <= pval_thresh)
    print(sum(idx), "gene perturbations in the map.")
    map_data = Bunch(features=map_data.features[idx], metadata=md[idx])
    np.random.seed(cst.RANDOM_SEED)
    # numpy requires seeds to be between 0 and 2 ** 32 - 1
    random_seed_pairs = np.random.randint(2**32, size=cst.RANDOM_COUNT * 2).reshape(cst.RANDOM_COUNT, 2)
    res = defaultdict(dict)  # type: ignore
    for rs1, rs2 in random_seed_pairs:
        res_seed = res[f"Seeds_{rs1}_{rs2}"]
        for src in benchmark_sources:
            if src not in res_seed:
                res_curr = compute_pairwise_metrics(
                    map_data,
                    src,
                    pert_label_col,
                    cst.RECALL_PERC_THR_PAIRS,
                    rs1,
                    rs2,
                    cst.N_NULL_SAMPLES,
                    cst.N_NULL_SAMPLES,
                )
                if res_curr is not None:
                    res_seed[src] = res_curr

        res[f"Seeds_{rs1}_{rs2}"] = res_seed
    return res
