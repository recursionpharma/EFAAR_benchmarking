from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scvi.model import SCVI
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

import efaar_benchmarking.constants as cst


def embed_by_scvi_anndata(adata, batch_col=cst.REPLOGLE_BATCH_COL, n_latent=128, n_hidden=256) -> np.ndarray:
    """
    Embed the input AnnData object using scVI.

    Args:
        adata (anndata.AnnData): The AnnData object to be embedded.
        batch_col (str): The batch key in the AnnData object. Default is "gem_group".
        n_latent (int): The number of latent dimensions. Default is 128.
        n_hidden (int): The number of hidden dimensions. Default is 256.

    Returns:
        numpy.ndarray: Embedding of the input data using scVI.
    """
    adata = adata.copy()
    SCVI.setup_anndata(adata, batch_key=batch_col)
    vae = SCVI(adata, n_hidden=n_hidden, n_latent=n_latent)
    vae.train(use_gpu=True)
    return vae.get_latent_representation()


def embed_by_pca_anndata(adata, n_latent=100) -> np.ndarray:
    """
    Embed the input data using principal component analysis (PCA).
    Note that the data is centered by the `pca` function prior to PCA transformation.

    Args:
        adata (AnnData): Annotated data matrix.
        n_latent (int): Number of principal components to use.

    Returns:
        numpy.ndarray: Embedding of the input data using PCA.
    """
    sc.pp.pca(adata, n_comps=n_latent)
    return adata.obsm["X_pca"]


def centerscale(features: np.ndarray, metadata: pd.DataFrame = None, plate_col: Optional[str] = None) -> np.ndarray:
    """
    Center and scale the input features.

    Args:
        features (np.ndarray): Input features to be centered and scaled.
        metadata (pd.DataFrame): Metadata information for the input features.
        plate_col (str): Name of the column in metadata that contains plate information.

    Returns:
        np.ndarray: Centered and scaled features.
    """
    if plate_col is None:
        features = StandardScaler().fit_transform(features)
    else:
        if metadata is None:
            raise ValueError("metadata must be provided if plate_col is not None")
        unq_plates = metadata[plate_col].unique()
        for plate in unq_plates:
            ind = metadata[plate_col] == plate
            features[ind, :] = StandardScaler().fit_transform(features[ind, :])
    return features


def embed_align_by_pca(
    features: np.ndarray,
    metadata: pd.DataFrame = None,
    variance_or_ncomp=100,
    plate_col: Optional[str] = None,
) -> np.ndarray:
    """
    Embed the input data using principal component analysis (PCA).
    Note that we explicitly center & scale the data by plate before and after calling `PCA`.
    Centering and scaling is done by plate if `plate_col` is not None, and on the whole data otherwise.
    Note that `PCA` transformer also does mean-centering on the whole data prior to the PCA operation.

    Args:
        features (np.ndarray): Features to transform
        metadata (pd.DataFrame): Metadata. Defaults to None.
        variance_or_ncomp (float, optional): Variance or number of components to keep after PCA.
            Defaults to 100 (n_components). If between 0 and 1, select the number of components such that
            the amount of variance that needs to be explained is greater than the percentage specified.
        plate_col (str, optional): Column name for plate metadata. Defaults to None.
    Returns:
        np.ndarray: Transformed data using PCA.
    """

    features = centerscale(features, metadata, plate_col)
    features = PCA(variance_or_ncomp).fit_transform(features)
    features = centerscale(features, metadata, plate_col)

    return features


def align_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    scale: bool = True,
    pert_col: str = cst.REPLOGLE_PERT_LABEL_COL,
    control_key: str = cst.CONTROL_PERT_LABEL,
) -> np.ndarray:
    """
    Center the embeddings by the control perturbation units in the metadata.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aligned.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        scale (bool): Whether to scale the embeddings besides centering. Defaults to True.
        pert_col (str, optional): The column in the metadata containing perturbation information.
            Defaults to cst.REPLOGLE_PERT_LABEL_COL.
        control_key (str, optional): The key for non-targeting controls in the metadata.
            Defaults to cst.CONTROL_PERT_LABEL.

    Returns:
        numpy.ndarray: The aligned embeddings.
    """
    ss = StandardScaler() if scale else StandardScaler(with_std=False)
    ss.fit(embeddings[metadata[pert_col].values == control_key])
    return ss.transform(embeddings)


def aggregate(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str = cst.REPLOGLE_PERT_LABEL_COL,
    control_key: str = cst.CONTROL_PERT_LABEL,
    method="mean",
) -> Bunch[pd.DataFrame, pd.DataFrame]:
    """
    Apply the mean or median aggregation to replicate embeddings for each perturbation.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aggregated.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        pert_col (str, optional): The column in the metadata containing perturbation information.
            Defaults to cst.REPLOGLE_PERT_LABEL_COL.
        control_key (str, optional): The key for non-targeting controls in the metadata.
            Defaults to cst.CONTROL_PERT_LABEL.
        method (str, optional): The aggregation method to use. Must be either "mean" or "median".
            Defaults to "mean".

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
        if method == "mean":
            final_embeddings[i, :] = np.mean(embeddings[idxs, :], axis=0)
        elif method == "median":
            final_embeddings[i, :] = np.median(embeddings[idxs, :], axis=0)
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
    return Bunch(features=pd.DataFrame(final_embeddings), metadata=pd.DataFrame.from_dict({pert_col: unique_perts}))


def filter_cpg16_crispr(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    filter_by_intensity: bool = True,
    filter_by_cell_count: bool = True,
    drop_image_cols: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the given features and metadata dataframes based on various criteria.

    Args:
        features (pd.DataFrame): A dataframe containing the features to filter.
        metadata (pd.DataFrame): A dataframe containing the metadata to filter.
        filter_by_intensity (bool, optional): Whether to filter by intensity. Defaults to True.
        filter_by_cell_count (bool, optional): Whether to filter by cell count. Defaults to True.
        drop_image_cols (bool, optional): Whether to drop image columns. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the filtered features and metadata dataframes.
    """
    if filter_by_intensity:
        features = features.loc[
            EllipticEnvelope(contamination=0.01, random_state=42).fit_predict(
                features[[col for col in features.columns if "ImageQuality_MeanIntensity" in col]]
            )
            == 1
        ]
    if filter_by_cell_count:
        mask = np.full(len(features), True)
        for colname in ["Cytoplasm_Number_Object_Number", "Nuclei_Number_Object_Number"]:
            mask = mask & (features[colname] >= 50) & (features[colname] <= 350)
        features = features.loc[mask]
    if drop_image_cols:
        features = features.drop(columns=[col for col in features.columns if col.startswith("Image_")])

    metadata_cols = metadata.columns
    merged_data = metadata.merge(features, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
    return merged_data.drop(columns=metadata_cols), merged_data[metadata_cols]
