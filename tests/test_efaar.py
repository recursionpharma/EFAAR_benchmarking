import numpy as np
import pandas as pd
from anndata import AnnData

from efaar_benchmarking import efaar


def test_embed_by_scvi_anndata():
    adata = AnnData(np.random.randint(low=0, high=100, size=(10, 300)))
    adata.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5
    embedding = efaar.embed_by_scvi_anndata(adata, "batch", 7, 14)
    assert embedding.shape == (10, 7)


def test_embed_by_pca_anndata():
    adata = AnnData(np.random.rand(10, 300))
    adata.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5
    embedding = efaar.embed_by_pca_anndata(adata, "batch", 7)
    assert embedding.shape == (10, 7)


def test_centerscale_on_batch():
    features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    metadata = pd.DataFrame({"batch": ["batch1", "batch1", "batch2"]})
    batch_col = "batch"
    expected_result = np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]])
    scaled_features = efaar.centerscale_by_batch(features, metadata, batch_col)
    assert np.array_equal(scaled_features, expected_result)


def test_centerscale_on_controls():
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    metadata = pd.DataFrame({"perturbation": ["control", "gene", "control"]})
    pert_col = "perturbation"
    control_key = "control"
    expected_result = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    aligned_embeddings = efaar.centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    assert np.array_equal(aligned_embeddings, expected_result)


def test_aggregate():
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    metadata = pd.DataFrame({"perturbation": ["control", "gene1", "gene1", "gene2"]})
    pert_col = "perturbation"
    control_key = "control"
    method = "mean"
    expected_features = pd.DataFrame([[5.5, 6.5, 7.5], [10, 11, 12]])
    expected_metadata = pd.DataFrame({"perturbation": ["gene1", "gene2"]})
    result = efaar.aggregate(embeddings, metadata, pert_col, control_key, method)
    assert np.allclose(result.features, expected_features, atol=1e-6)
    assert result.metadata.equals(expected_metadata)


def test_filter_cell_profiler_features():
    features = pd.DataFrame(
        {
            "Image_Granularity_12_ER": np.random.rand(10),
            "Cytoplasm_Number_Object_Number": np.linspace(30, 60, 10),
            "Nuclei_Number_Object_Number": np.linspace(25, 55, 10),
        }
    )
    metadata = pd.DataFrame({"id": range(10)})
    filtered_features, _ = efaar.filter_cell_profiler_features(features, metadata)
    assert not filtered_features.empty
    assert "Image_Granularity_12_ER" not in filtered_features.columns
