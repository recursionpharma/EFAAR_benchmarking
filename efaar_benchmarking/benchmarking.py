from efaar_benchmarking.utils import compute_pairwise_metrics, get_feats_w_indices
import efaar_benchmarking.constants as cst
from sklearn.utils import Bunch
import numpy as np
from collections import defaultdict


def pert_stats(
    map_data: Bunch,
    filter_on_pert_type=False,
    filter_on_well_type=False,
    pert_sig_thr: float = cst.PERT_SIG_PVAL_THR,
):
    md = map_data.metadata
    idx = [True] * len(md)
    if filter_on_pert_type:
        idx = idx & (md[cst.PERT_TYPE_COL] == cst.PERT_TYPE)
    if filter_on_well_type:
        idx = idx & (md[cst.WELL_TYPE_COL] == cst.WELL_TYPE)
    pidx = md[cst.PERT_SIG_PVAL_COL] <= pert_sig_thr
    return {
        "all_pert_count": sum(idx),
        "pp_pert_count": sum(idx & pidx),
        "pp_pert_percent": sum(idx & pidx) / sum(idx),
    }


def benchmark(
    map_data: Bunch,
    pert_label_col: str = cst.PERT_LABEL_COL,
    benchmark_sources: list = cst.BENCHMARK_SOURCES,
    filter_on_pert_type: bool = False,
    filter_on_well_type: bool = False,
    filter_on_pert_prints: bool = False,
    run_count: int = cst.RANDOM_COUNT,
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
    res = defaultdict(dict)  # type: ignore
    feats_w_indices = get_feats_w_indices(map_data, pert_label_col)
    if len(set(feats_w_indices.index)) >= cst.MIN_REQ_ENT_CNT:
        np.random.seed(cst.RANDOM_SEED)
        # numpy requires seeds to be between 0 and 2 ** 32 - 1
        random_seed_pairs = np.random.randint(2**32, size=run_count * 2).reshape(run_count, 2)
        for rs1, rs2 in random_seed_pairs:
            res_seed = res[f"Seeds_{rs1}_{rs2}"]
            for src in benchmark_sources:
                if src not in res_seed:
                    res_curr = compute_pairwise_metrics(
                        feats_w_indices,
                        src,
                        cst.RECALL_PERC_THR_PAIR,
                        rs1,
                        rs2,
                        cst.N_NULL_SAMPLES,
                        cst.N_NULL_SAMPLES,
                    )
                    if res_curr is not None:
                        res_seed[src] = res_curr

            res[f"Seeds_{rs1}_{rs2}"] = res_seed
    else:
        print("Not enough entities in the map for benchmarking")
    return res
