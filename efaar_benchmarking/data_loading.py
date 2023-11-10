import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scanpy as sc
import wget


def load_cpg16_crispr(data_path: str = "data/") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return the JUMP-CP (cpg0016) CRISPR dataset.
    The metadata is downloaded from here:
        https://zenodo.org/records/7661296/files/jump-cellpainting/metadata-v0.5.0.zip?download=1
    The cellprofiler features are downloaded from here:
        https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0016-jump/
    We read the metadata first, filter it to CRISPR plates, and download the features for these plates only.

    Parameters:
    data_path (str): Path to the directory containing the dataset files.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
        - features: A DataFrame containing the CRISPR dataset features.
        - metadata: A DataFrame containing the CRISPR dataset metadata.
    """
    metadata_source_path = "https://zenodo.org/records/7661296/files/jump-cellpainting/datasets-v0.5.0.zip?download=1"
    plate_file_name = "plate.csv.gz"
    well_file_name = "well.csv.gz"
    crispr_file_name = "crispr.csv.gz"
    plate_file_path = os.path.join(data_path, plate_file_name)
    well_file_path = os.path.join(data_path, well_file_name)
    crispr_file_path = os.path.join(data_path, crispr_file_name)
    if not (os.path.exists(plate_file_path) and os.path.exists(well_file_path) and os.path.exists(crispr_file_path)):
        
        path_to_zip_file = data_path + "tmp.zip"
        wget.download(metadata_source_path, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            for name in zip_ref.namelist():
                if name.endswith(plate_file_name) or name.endswith(well_file_name) or name.endswith(crispr_file_name):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_ref.extract(name, temp_dir)
                        shutil.move(os.path.join(temp_dir, name), os.path.join(data_path, os.path.basename(name)))
        os.remove(path_to_zip_file)

    plates = pd.read_csv(plate_file_path)
    crispr_plates = plates.query("Metadata_PlateType=='CRISPR'")
    wells = pd.read_csv(well_file_path)
    well_plate = wells.merge(crispr_plates, on=["Metadata_Source", "Metadata_Plate"])
    crispr = pd.read_csv(crispr_file_path)
    metadata = well_plate.merge(crispr, on="Metadata_JCP2022")

    cp_feature_source_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/profiles/"
        "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    )

    features_file_path = os.path.join(data_path, "cpg_features.parquet")
    if not os.path.exists(features_file_path):

        def load_plate_features(path: str):
            return pd.read_parquet(path, storage_options={"anon": True})

        cripsr_plates = metadata[
            ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_PlateType"]
        ].drop_duplicates()
        features = []
        with ThreadPoolExecutor(max_workers=10) as executer:
            future_to_plate = {
                executer.submit(
                    load_plate_features, cp_feature_source_formatter.format(**row.to_dict())
                ): cp_feature_source_formatter.format(**row.to_dict())
                for _, row in cripsr_plates.iterrows()
            }
            for future in as_completed(future_to_plate):
                features.append(future.result())
        pd.concat(features).to_parquet(features_file_path)
    features = pd.read_parquet(features_file_path).dropna(axis=1)
    return features, metadata


def load_replogle(gene_type: str, data_type: str, data_path: str = "data/") -> sc.AnnData:
    """
    Load Replogle et al. 2022 single-cell RNA-seq data for K562 cells  published here: https://pubmed.ncbi.nlm.nih.gov/35688146/
    Four types of K562 data and downloaded using the links at:
    plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387

    Parameters:
    gene_type (str): Type of genes to load. Must be either 'essential' or 'genome_wide'.
    data_type (str): Type of data to load. Must be either 'raw' or 'normalized'.
        Normalized means Z-normalized by gemgroup.
    data_path (str): Path to the directory where the data will be downloaded and saved.

    Returns:
    Anndata object containing the single-cell RNA-seq data.
    """
    if gene_type == "essential":
        if data_type == "raw":
            filename = "K562_essential_raw_singlecell_01.h5ad"
            src = "https://ndownloader.figshare.com/files/35773219"
        elif data_type == "normalized":
            filename = "K562_essential_normalized_singlecell_01.h5ad"
            src = "https://ndownloader.figshare.com/files/35773075"
        else:
            raise ValueError("data_type must be either raw or normalized")

    elif gene_type == "genome_wide":
        if data_type == "raw":
            filename = "K562_gwps_raw_singlecell_01.h5ad"
            src = "https://ndownloader.figshare.com/files/35775507"
        elif data_type == "normalized":
            filename = "K562_gwps_normalized_singlecell_01.h5ad"
            src = "https://ndownloader.figshare.com/files/35773217"
        else:
            raise ValueError("data_type must be either raw or normalized")

    else:
        raise ValueError("gene_type must be either essential or genome_wide")

    if not os.path.exists(data_path + filename):
        wget.download(src, data_path + filename)

    adata = sc.read_h5ad(data_path + filename)
    adata.X = np.nan_to_num(adata.X)
    return adata
