import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from efaar_benchmarking import benchmarking, constants
from efaar_benchmarking.benchmarking import (
    AggregateBy,
    AverageType,
    BenchmarkConfig,
    compound_gene_benchmark,
    compute_metrics,
    compute_similarities,
    process_predictions,
    sample_for_item,
)


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


@pytest.fixture
def sample_truth_data():
    """Create sample ground truth data with known properties."""
    return pd.DataFrame(
        {
            "treatment": ["compound1", "compound1", "compound2", "compound2"],
            "gene_symbol": ["gene1", "gene2", "gene1", "gene2"],
            "nM_value": [100, 5000, 15000, 500],  # Mix of active, gray zone, and inactive
        }
    )


@pytest.fixture
def sample_map_data():
    """Create sample embedding data with known similarities."""
    features = pd.DataFrame(
        {"feat1": [1.0, 0.0, 0.0, 1.0], "feat2": [0.0, 1.0, 1.0, 0.0]},
        index=["compound1_id", "compound2_id", "gene1_id", "gene2_id"],
    )

    metadata = pd.DataFrame(
        {
            "perturbation": ["compound1", "compound2", "gene1", "gene2"],
            "concentration": ["1.0", "1.0", np.nan, np.nan],
        },
        index=["compound1_id", "compound2_id", "gene1_id", "gene2_id"],
    )

    return Bunch(features=features, metadata=metadata)


def test_compute_similarities_basic(sample_map_data):
    """Test basic similarity computation."""
    truth = pd.DataFrame({"treatment": ["compound1", "compound2"], "gene_symbol": ["gene1", "gene2"]})

    sims = compute_similarities(truth, sample_map_data, "perturbation")

    assert isinstance(sims, pd.DataFrame)
    assert sims.shape == (2, 2)  # 2 compounds x 2 genes
    # compound1 should be perfectly similar to gene1 (same features)
    assert np.isclose(sims.loc[("compound1", "1.0"), "gene1"], 1.0)


def test_compute_similarities_randomized(sample_map_data):
    """Test randomized similarity computation."""
    truth = pd.DataFrame({"treatment": ["compound1"], "gene_symbol": ["gene1"]})

    sims = compute_similarities(truth, sample_map_data, "perturbation", randomize=True)

    assert isinstance(sims, pd.DataFrame)
    assert sims.shape == (1, 1)
    assert 0 <= sims.iloc[0, 0] <= 1


def test_sample_for_item():
    """Test negative sampling logic."""
    item_data = pd.DataFrame({"gene_symbol": ["gene1", "gene2", "gene3"], "nM_value": [100, 5000, 15000]})
    pool = {"gene1", "gene2", "gene3", "gene4", "gene5"}

    items, labels = sample_for_item(
        item_data,
        pool,
        activity_threshold=1000,
        inactivity_threshold=10000,
        target_col="gene_symbol",
        min_negatives=2,
        random_seed=42,
    )

    assert len(items) > 0
    assert len(labels) == len(items)
    assert sum(labels) == 1  # Only gene1 should be positive
    assert "gene3" not in items  # Should be excluded as it's in gray zone


def test_compute_metrics():
    """Test metric computation with known values."""
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 0, 0, 1])

    ap, auc = compute_metrics(scores, labels)

    assert 0 <= ap <= 1
    assert 0 <= auc <= 1
    # With these specific values, AP should be less than 0.75
    assert ap < 0.75


def test_full_benchmark_macro_compound(sample_truth_data, sample_map_data):
    """Test full benchmark with macro averaging by compound."""
    config = BenchmarkConfig(
        average_type=AverageType.MACRO, aggregate_by=AggregateBy.COMPOUND, min_negatives=2, random_seed=42
    )

    results = compound_gene_benchmark(
        map_data=sample_map_data,
        activity_threshold=1000,
        inactivity_threshold=10000,
        truth_data=sample_truth_data,
        config=config,
    )

    assert isinstance(results, pd.DataFrame)
    assert "concentration" in results.columns
    assert "average_precision" in results.columns
    assert "auc_roc" in results.columns
    assert "baseline_average_precision" in results.columns
    assert "baseline_auc_roc" in results.columns
    assert len(results) > 0
    assert "max" in results["concentration"].values


def test_full_benchmark_micro_gene(sample_truth_data, sample_map_data):
    """Test full benchmark with micro averaging by gene."""
    config = BenchmarkConfig(
        average_type=AverageType.MICRO, aggregate_by=AggregateBy.GENE, min_negatives=2, random_seed=42
    )

    results = compound_gene_benchmark(
        map_data=sample_map_data,
        activity_threshold=1000,
        inactivity_threshold=10000,
        truth_data=sample_truth_data,
        config=config,
    )

    assert isinstance(results, pd.DataFrame)
    assert all(results["baseline_auc_roc"] == 0.5)  # Random baseline should have 0.5 AUC-ROC


def test_benchmark_edge_cases(sample_map_data):
    """Test benchmark behavior with edge cases."""
    # Empty truth data
    empty_truth = pd.DataFrame(columns=["treatment", "gene_symbol", "nM_value"])
    with pytest.raises(ValueError):
        compound_gene_benchmark(map_data=sample_map_data, truth_data=empty_truth)

    # All inactive data
    all_inactive = pd.DataFrame(
        {
            "treatment": ["compound1"],
            "gene_symbol": ["gene1"],
            "nM_value": [20000],  # Above inactivity threshold
        }
    )
    results = compound_gene_benchmark(map_data=sample_map_data, truth_data=all_inactive)
    assert len(results) > 0
    assert all(results["average_precision"] == 0.0)  # No positives should give 0 AP


def test_process_predictions(sample_truth_data, sample_map_data):
    """Test prediction processing logic."""
    config = BenchmarkConfig()
    similarities = compute_similarities(sample_truth_data, sample_map_data, "perturbation")

    predictions = process_predictions(
        sample_truth_data, similarities, config, thresholds=(1000, 10000), pert_col="perturbation"
    )

    assert isinstance(predictions, dict)
    assert "max" in predictions
    assert all(isinstance(p, list) for p in predictions.values())

    # Check prediction format
    for conc, preds in predictions.items():
        if preds:
            scores, labels = preds[0]
            assert isinstance(scores, np.ndarray)
            assert isinstance(labels, np.ndarray)
            assert len(scores) == len(labels)
            assert set(labels).issubset({0, 1})


def test_config_validation():
    """Test configuration validation."""
    # Invalid average type
    with pytest.raises(ValueError):
        BenchmarkConfig(average_type="invalid")

    # Invalid aggregate by
    with pytest.raises(ValueError):
        BenchmarkConfig(aggregate_by="invalid")

    # Invalid min negatives
    with pytest.raises(ValueError):
        BenchmarkConfig(min_negatives=-1)
