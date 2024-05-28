import numpy as np
import pandas as pd

from efaar_benchmarking import benchmarking


def test_generate_query_cossims():
    feats = pd.DataFrame(
        np.array([[1, 4, 9], [2, 5, 8], [9, 5, 7]]),
        index=[f"g{i}" for i in range(3)],
        columns=[f"f{i}" for i in range(3)],
    )
    gt_source_df = pd.DataFrame({"entity1": ["g0", "g1"], "entity2": ["g1", "g2"], "source": ["s0", "s1"]})
    result = benchmarking.generate_query_cossims(feats, gt_source_df, min_req_entity_cnt=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, np.array([0.98463063, 0.82457065]), decimal=8)


def test_compute_process_cosine_sim_same_entities():
    feats1 = pd.DataFrame(
        np.random.rand(10, 3), index=[f"g{i}" for i in range(10)], columns=[f"f{i}" for i in range(3)]
    )
    feats2 = pd.DataFrame(
        np.random.rand(10, 3), index=[f"g{i}" for i in range(10)], columns=[f"f{i}" for i in range(3)]
    )
    result = benchmarking.compute_process_cosine_sim(feats1, feats2)
    assert result.shape == (90,)


def test_compute_process_cosine_sim_different_entities():
    feats1 = pd.DataFrame(
        np.random.rand(10, 3), index=[f"g{i}" for i in range(10)], columns=[f"f{i}" for i in range(3)]
    )
    feats2 = pd.DataFrame(
        np.random.rand(10, 3), index=[f"g{i}" for i in range(10, 20)], columns=[f"f{i}" for i in range(3)]
    )
    result = benchmarking.compute_process_cosine_sim(feats1, feats2)
    assert result.shape == (100,)


def test_compute_recall():
    null_distribution = np.array([1, 2, 3, 4, 5])
    query_distribution = np.array([1, 5])
    recall_threshold_pairs = [(0.1, 0.9), (0.2, 0.8)]
    expected_metrics = {
        "null_distribution_size": 5,
        "query_distribution_size": 2,
        "recall_0.1_0.9": 0.0,
        "recall_0.2_0.8": 1.0,
    }
    metrics = benchmarking.compute_recall(null_distribution, query_distribution, recall_threshold_pairs)
    assert metrics == expected_metrics


def test_filter_relationships():
    df = pd.DataFrame({"entity1": ["A", "B", "A", "C", "D"], "entity2": ["B", "A", "A", "D", "C"]})
    filtered_df = benchmarking.filter_relationships(df)
    assert len(filtered_df) == 2
