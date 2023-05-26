from sklearn.utils import Bunch
from sklearn.metrics.pairwise import cosine_similarity
import efaar_benchmarking.constants as cst
import numpy as np
import pandas as pd
from typing import Optional


def get_feats_w_indices(data: Bunch, pert_label_col: str) -> pd.DataFrame:
    return data.features.set_index(data.metadata[pert_label_col]).rename_axis(index=None)


def compute_process_cosine_sim(
    entity1_feats: pd.DataFrame,
    entity2_feats: pd.DataFrame,
    filter_to_pairs: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    cosi = pd.DataFrame(
        cosine_similarity(entity1_feats, entity2_feats), index=entity1_feats.index, columns=entity2_feats.index
    )
    # convert pairwise cosine similarity matrix to a data frame of triples so that filtering and grouping is easy
    cosi = cosi.stack()[np.ones(cosi.size).astype("bool")].reset_index()
    cosi.columns = ["entity1", "entity2", "cosine_sim"]  # type: ignore
    if filter_to_pairs is not None:
        cosi = cosi.merge(filter_to_pairs, how="right", on=["entity1", "entity2"])
    cosi = cosi[cosi.entity1 != cosi.entity2]  # remove self cosine similarities
    return cosi.cosine_sim.values


def generate_null_cossims(
    entity1_feats: pd.DataFrame,
    entity2_feats: pd.DataFrame,
    rseed_entity1: int,
    rseed_entity2: int,
    n_entity1: int,
    n_entity2: int,
) -> np.ndarray:
    np.random.seed(rseed_entity1)
    entity1_feats = entity1_feats.loc[np.random.choice(list(entity1_feats.index.unique()), n_entity1)]
    np.random.seed(rseed_entity2)
    entity2_feats = entity2_feats.loc[np.random.choice(list(entity2_feats.index.unique()), n_entity2)]
    return compute_process_cosine_sim(entity1_feats, entity2_feats)


def generate_query_cossims(
    entity1_feats: pd.DataFrame,
    entity2_feats: pd.DataFrame,
    gt_source_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    gt_source_df = gt_source_df[
        gt_source_df.entity1.isin(entity1_feats.index) & gt_source_df.entity2.isin(entity2_feats.index)
    ]
    entity1_feats = entity1_feats.loc[list(set(gt_source_df.entity1))]
    entity2_feats = entity2_feats.loc[list(set(gt_source_df.entity2))]
    print("Took the overlap between the annotation source and the map entities.")
    if len(set(entity1_feats.index)) >= cst.MIN_REQ_ENT_CNT and len(set(entity2_feats.index)) >= cst.MIN_REQ_ENT_CNT:
        return compute_process_cosine_sim(entity1_feats, entity2_feats, gt_source_df)
    else:
        print("Not enough entities for benchmarking.")
        return None


def get_benchmark_data(src):
    return pd.read_csv(cst.BENCHMARK_DATA_DIR.joinpath(src + ".txt"))


def compute_pairwise_metrics(
    feats_w_indices: pd.DataFrame,
    src: str,
    thr_pair: tuple,
    rseed_ent1: int,
    rseed_ent2: int,
    num_null_samp_ent1: int,
    num_null_samp_ent2: int,
) -> Optional[dict]:
    print("Running benchmarking for", src)
    df_bm = generate_query_cossims(feats_w_indices, feats_w_indices, get_benchmark_data(src))
    if df_bm is not None:
        df_null = generate_null_cossims(
            feats_w_indices,
            feats_w_indices,
            rseed_ent1,
            rseed_ent2,
            num_null_samp_ent1,
            num_null_samp_ent2,
        )
        res = {"gt": df_bm}
        gt_perc = np.searchsorted(np.sort(df_null), df_bm) / len(df_null)
        l_thr = np.min(thr_pair)
        r_thr = np.max(thr_pair)
        res["recall"] = sum((gt_perc <= l_thr) | (gt_perc >= r_thr)) / len(gt_perc)
        return res
    else:
        return None


def get_benchmark_metrics(bm_res: dict) -> pd.DataFrame:
    bm_sources = list(list(bm_res.values())[0].keys())
    recall_vals = [np.mean([v[src]["recall"] for k, v in bm_res.items()]) for src in bm_sources]
    return pd.DataFrame({"source": bm_sources, "recall": recall_vals})
