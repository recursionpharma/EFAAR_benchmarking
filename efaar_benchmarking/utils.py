from sklearn.utils import Bunch
from sklearn.metrics.pairwise import cosine_similarity
import efaar_benchmarking.constants as cst
import numpy as np
import pandas as pd
from typing import Optional


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
        np.ndarray: A NumPy array containing the null cosine similarity values between the randomly sampled subsets of entities.
    """

    np.random.seed(rseed_entity1)
    entity1_feats = feats.loc[np.random.choice(list(feats.index.unique()), n_sample_entities)]
    np.random.seed(rseed_entity2)
    entity2_feats = feats.loc[np.random.choice(list(feats.index.unique()), n_sample_entities)]
    return compute_process_cosine_sim(entity1_feats, entity2_feats)


def generate_query_cossims(
    feats: pd.DataFrame,
    gt_source_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Generate query-specific cosine similarity values between subsets of entities' features.

    Args:
        feats (pd.DataFrame): Features of the first set of entities.
        gt_source_df (pd.DataFrame): DataFrame containing ground truth annotation sources.

    Returns:
        Optional[np.ndarray]: A NumPy array containing the query-specific cosine similarity values, or None
            if there are not enough entities for benchmarking.
    """

    gt_source_df = gt_source_df[
        gt_source_df.entity1.isin(feats.index) & gt_source_df.entity2.isin(feats.index)
    ]
    entity1_feats = feats.loc[list(set(gt_source_df.entity1))]
    entity2_feats = feats.loc[list(set(gt_source_df.entity2))]
    if len(set(entity1_feats.index)) >= cst.MIN_REQ_ENT_CNT and len(set(entity2_feats.index)) >= cst.MIN_REQ_ENT_CNT:
        return compute_process_cosine_sim(entity1_feats, entity2_feats, gt_source_df)
    else:
        return None


def get_benchmark_data(src):
    """Load benchmark data from a text file.

    Args:
        src (str): The source identifier for the benchmark data.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark data loaded from the text file.
    """

    return pd.read_csv(cst.BENCHMARK_DATA_DIR.joinpath(src + ".txt"))


def compute_recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
    recall_threshold_pairs: list,
) -> dict:
    """Compute recall at given percentage thresholds for a query distribution with respect to a null distribution. 
    Each recall threshold is a pair of floats (left, right) where left and right are floats between 0 and 1.

    Parameters:
        null_distribution (np.ndarray): The null distribution to compare against
        query_distribution (np.ndarray): The query distribution
        recall_threshold_pairs (list) A list of pairs of floats (left, right) that represent different recall threshold pairs, where
            left and right are floats between 0 and 1.

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
    query_percentage_ranks = np.searchsorted(sorted_null_distribution, query_distribution) / len(
        sorted_null_distribution
    )
    for threshold_pair in recall_threshold_pairs:
        left_threshold, right_threshold = np.min(threshold_pair), np.max(threshold_pair)
        metrics[f"recall_{left_threshold}_{right_threshold}"] = sum(
            (query_percentage_ranks <= left_threshold) | (query_percentage_ranks >= right_threshold)
        ) / len(query_percentage_ranks)
    return metrics
