import numpy as np
import pandas as pd
import scanpy as sc
from scvi.model import SCVI
from sklearn.utils import Bunch

import efaar_benchmarking.constants as cst


def embed_by_scvi(adata, batch_key="gem_group", n_latent=128, n_hidden=256):
    """
    Embeds the input AnnData object using scVI.

    Parameters:
    adata (anndata.AnnData): The AnnData object to be embedded.
    batch_key (str): The batch key in the AnnData object. Default is "gem_group".
    n_latent (int): The number of latent dimensions. Default is 128.
    n_hidden (int): The number of hidden dimensions. Default is 256.

    Returns:
    None
    """
    SCVI.setup_anndata(adata, batch_key=batch_key)
    vae = SCVI(adata, n_hidden=n_hidden, n_latent=n_latent)
    vae.train(use_gpu=True)
    return vae.get_latent_representation()


def embed_by_pca(adata, n_latent=100):
    """
    Embeds the input data using PCA.

    Parameters:
    adata (AnnData): Annotated data matrix.
    n_latent (int): Number of principal components to use.

    Returns:
    numpy.ndarray: Embedding of the input data using PCA.
    """
    sc.pp.pca(adata, n_comps=n_latent)
    return adata.obsm["X_pca"]


def align_by_centering(embeddings, metadata, control_key=cst.CONTROL_PERT_LABEL, pert_col=cst.PERT_LABEL_COL):
    """
    Applies the centerscale method to align embeddings based on the centering perturbations in the metadata.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aligned.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        control_key (str, optional): The key for non-targeting controls in the metadata.
            Defaults to cst.CONTROL_PERT_LABEL.
        pert_col (str, optional): The column in the metadata containing perturbation information.
            Defaults to cst.PERT_LABEL_COL.

    Returns:
        numpy.ndarray: The aligned embeddings.
    """
    ntc_idxs = np.where(metadata[pert_col].values == control_key)[0]
    ntc_center = embeddings[ntc_idxs].mean(0)
    return embeddings - ntc_center


def aggregate_by_mean(embeddings, metadata, control_key=cst.CONTROL_PERT_LABEL, pert_col=cst.PERT_LABEL_COL):
    """
    Applies the mean aggregation to aggregate replicate embeddings for each perturbation.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aggregated.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        control_key (str, optional): The key for non-targeting controls in the metadata. Defaults to "non-targeting".
        PERT_COL (str, optional): The column in the metadata containing perturbation information. Defaults to "gene".

    Returns:
        Bunch: A named tuple containing two pandas DataFrames:
            - 'features': The aggregated embeddings.
            - 'metadata': A DataFrame containing the perturbation labels for each row in 'data'.
    """
    unique_perts = list(np.unique(metadata[pert_col].values))
    unique_perts.remove(control_key)
    final_embeddings = np.zeros((len(unique_perts), embeddings.shape[1]))
    for i, pert in enumerate(unique_perts):
        idxs = np.where(metadata[pert_col].values == pert)[0]
        final_embeddings[i, :] = embeddings[idxs, :].mean(0)
    return Bunch(
        features=pd.DataFrame(final_embeddings), metadata=pd.DataFrame.from_dict({pert_col: unique_perts})
    )
