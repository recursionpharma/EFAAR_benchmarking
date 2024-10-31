import os

import numpy as np
import pandas as pd
import pytest

from efaar_benchmarking import benchmarking, constants


@pytest.fixture
def sample_map_data():
    data = {
        "perturbation": ["compound1", "gene1", "compound2", "gene2"],
        "concentration": [10.0, np.nan, 1.0, np.nan],
        "feature_1": [0.1, 0.2, 0.3, 0.4],
        "feature_2": [0.5, 0.6, 0.7, 0.8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_read_csv(monkeypatch):
    def mock_read_csv(path):
        data = {
            "treatment": ["compound1", "compound2"],
            "gene_symbol": ["gene1", "gene2"],
            "nM_value": [500, 1500],
        }
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)


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


@pytest.mark.parametrize(
    "compound, gene, concentration, expected",
    [
        ("compound1", "gene1", 10.0, 0.9923),
        ("compound2", "gene2", 1.0, 0.9983),
        ("compound1", "gene3", 10.0, None),
    ],
)
def test_cosine_similarity_from_map(compound, gene, concentration, expected, sample_map_data):
    result = benchmarking.cosine_similarity_from_map(compound, gene, concentration, sample_map_data)
    print(result)
    if result is not None:
        assert np.isclose(result, expected, atol=1e-4)
    else:
        assert result == expected


def test_compound_gene_benchmark(mock_read_csv, sample_map_data):
    aps_df, curves = benchmarking.compound_gene_benchmark(
        sample_map_data, nM_activity_threshold=1000, benchmark_data_dir="dummy_dir"
    )

    assert not aps_df.empty
    assert list(aps_df.columns) == ["concentration", "average_precision"]
    assert "1.0" in curves
    assert "max" in curves
    assert isinstance(curves["max"], tuple)
