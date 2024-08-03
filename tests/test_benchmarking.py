import os

import numpy as np
import pandas as pd

from efaar_benchmarking import benchmarking, constants


def test_pert_signal_consistency_metric():
    arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    null = np.array([1, 2, 3, 4, 5])

    result = benchmarking.pert_signal_consistency_metric(arr1)
    assert result == 1

    result = benchmarking.pert_signal_consistency_metric(arr1, null)
    assert round(result[0]) == 1
    assert round(result[1]) == 1


def test_pert_signal_magnitude_metric():
    arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    arr2 = np.array([[6, 7, 8, 9, 10], [6, 7, 8, 9, 10], [6, 7, 8, 9, 10], [6, 7, 8, 9, 10], [6, 7, 8, 9, 10]])
    null = np.array([1, 2, 3, 4, 5])

    result = benchmarking.pert_signal_magnitude_metric(arr1, arr1)
    assert result == 0

    result = benchmarking.pert_signal_magnitude_metric(arr1, arr2)
    assert round(result) == 22

    result = benchmarking.pert_signal_magnitude_metric(arr1, arr2, null)
    assert round(result[0]) == 22
    assert round(result[1]) == 0


def test_benchmark_annotations():
    benchmark_files = [f for f in os.listdir(constants.BENCHMARK_DATA_DIR) if f.endswith(".txt")]
    for file in benchmark_files:
        with open(os.path.join(constants.BENCHMARK_DATA_DIR, file)) as f:
            header = f.readline().strip().split(",")
            assert header[0] == "entity1" and header[1] == "entity2"


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
