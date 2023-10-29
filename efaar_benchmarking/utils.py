from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import efaar_benchmarking.constants as cst


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
    Filters a DataFrame of relationships between entities, removing any rows with self-relationships, ie.
        where the same entity appears in both columns.

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


def get_benchmark_relationships(benchmark_data_dir: str, src: str):
    """
    Reads a CSV file containing benchmark data and returns a filtered DataFrame.

    Args:
        benchmark_data_dir (str): The directory containing the benchmark data files.
        src (str): The name of the source containing the benchmark data.

    Returns:
        pd.DataFrame: A filtered DataFrame containing the benchmark data.
    """
    return filter_relationships(pd.read_csv(Path(benchmark_data_dir).joinpath(src + ".txt")))


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
