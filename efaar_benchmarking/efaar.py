import numpy as np
import pandas as pd
import efaar_benchmarking.constants as cst
from scvi.model import SCVI
import scanpy as sc
from sklearn.utils import Bunch


def embed_by_scvi(adata, BATCH_KEY="gem_group", N_LATENT=128, N_HIDDEN=256):
    """
    Embeds the input AnnData object using scVI.

    Parameters:
    adata (anndata.AnnData): The AnnData object to be embedded.
    BATCH_KEY (str): The batch key in the AnnData object. Default is "gem_group".
    N_LATENT (int): The number of latent dimensions. Default is 128.
    N_HIDDEN (int): The number of hidden dimensions. Default is 256.

    Returns:
    None
    """
    SCVI.setup_anndata(adata, batch_key=BATCH_KEY)
    vae = SCVI(adata, n_hidden=N_HIDDEN, n_latent=N_LATENT)
    vae.train(use_gpu=True)
    return vae.get_latent_representation()


def embed_by_pca(adata, N_LATENT=100):
    """
    Embeds the input data using PCA.

    Parameters:
    adata (AnnData): Annotated data matrix.
    BATCH_KEY (str): Key for batch information in adata.obs.
    N_LATENT (int): Number of principal components to use.

    Returns:
    numpy.ndarray: Embedding of the input data using PCA.
    """
    sc.pp.pca(adata, n_comps=N_LATENT)
    return adata.obsm["X_pca"]


def align_by_centering(embeddings, metadata, NTC_KEY="non-targeting", PERT_COL="gene"):
    """
    Applies the centerscale method to align embeddings based on the centering perturbations in the metadata.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aligned.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        NTC_KEY (str, optional): The key for non-targeting controls in the metadata. Defaults to "non-targeting".
        PERT_COL (str, optional): The column in the metadata containing perturbation information. Defaults to "gene".

    Returns:
        numpy.ndarray: The aligned embeddings.
    """
    ntc_idxs = np.where(metadata[PERT_COL].values == NTC_KEY)[0]
    ntc_center = embeddings[ntc_idxs].mean(0)
    return embeddings - ntc_center


def aggregate_by_mean(embeddings, metadata, NTC_KEY="non-targeting", PERT_COL="gene"):
    """
    Applies the mean aggregation to aggregate replicate embeddings for each perturbation.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aggregated.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        NTC_KEY (str, optional): The key for non-targeting controls in the metadata. Defaults to "non-targeting".
        PERT_COL (str, optional): The column in the metadata containing perturbation information. Defaults to "gene".

    Returns:
        Bunch: A named tuple containing two pandas DataFrames:
            - 'features': The aggregated embeddings.
            - 'metadata': A DataFrame containing the perturbation labels for each row in 'data'.
    """
    unique_perts = list(np.unique(metadata[PERT_COL].values))
    unique_perts.remove(NTC_KEY)
    final_embeddings = np.zeros((len(unique_perts), embeddings.shape[1]))
    for i, pert in enumerate(unique_perts):
        idxs = np.where(metadata[PERT_COL].values == pert)[0]
        final_embeddings[i, :] = embeddings[idxs, :].mean(0)
    return Bunch(
        features=pd.DataFrame(final_embeddings), metadata=pd.DataFrame.from_dict({cst.PERT_LABEL_COL: unique_perts})
    )
