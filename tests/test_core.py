import numpy as np
import pandas as pd

from efaar_benchmarking.benchmarking import compute_recall, generate_query_cossims


def test_generate_query_cossims():
    feats = pd.DataFrame(
        {"feat1": [1, 2, 3], "feat2": [4, 5, 6], "feat3": [7, 8, 9]}, index=["entity1", "entity2", "entity3"]
    )

    gt_source_df = pd.DataFrame(
        {"entity1": ["entity1", "entity2"], "entity2": ["entity2", "entity3"], "source": ["source1", "source2"]}
    )

    result = generate_query_cossims(feats, gt_source_df, min_req_entity_cnt=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


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
    metrics = compute_recall(null_distribution, query_distribution, recall_threshold_pairs)
    assert metrics == expected_metrics
