import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scanpy as sc
import wget

from efaar_benchmarking.constants import PERISCOPE_BATCH_COL


def load_periscope(cell_type="HeLa", plate_type="DMEM", normalized=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PERISCOPE (cpg0021) data for a specific cell type.
    Find more information about the dataset here: https://www.biorxiv.org/content/10.1101/2023.08.06.552164v1
    The files containing metadata and CellProfiler features are downloaded from here:
        https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0021-periscope/

    Parameters:
    cell_type (str, optional): The cell type to load data for. Defaults to "HeLa".
    normalized (bool, optional): Whether to load normalized data. Defaults to True.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames named `features` and `metadata`,
        representing the loaded data.
    """
    per_data_all = []
    cp_feature_source_formatter = "s3://cellpainting-gallery/cpg0021-periscope/broad/workspace/profiles/{cell_type}/"
    if cell_type == "A549":
        plates_all = {"DMEM": ["A", "B", "C", "D", "E", "F", "G", "H", "N"]}
        if normalized:
            filename_formatter = "20200805_A549_WG_Screen_guide_normalized_ALLBATCHES___CP186{plate}___ALLWELLS.csv.gz"
        else:
            filename_formatter = "20200805_A549_WG_Screen_guide_ALLBATCHES___CP186{plate}___ALLWELLS.csv.gz"
    elif cell_type == "HeLa":
        plates_all = {"DMEM": ["A", "B", "D", "F", "H"], "HPLM": ["J", "K", "L", "N"]}
        if normalized:
            filename_formatter = "20210422_6W_CP257_guide_normalized_ALLBATCHES___CP257{plate}___ALLWELLS.csv.gz"
        else:
            filename_formatter = "20210422_6W_CP257_guide_ALLBATCHES___CP257{plate}___ALLWELLS.csv.gz"
    else:
        raise ValueError("cell_type must be either HeLa or A549")
    plates = plates_all.get(plate_type)
    if plates is None:
        raise ValueError(f"plate_type must be in {list(plates_all.keys())} for {cell_type}")

    cp_feature_source_formatter += filename_formatter
    with ThreadPoolExecutor(max_workers=10) as executer:
        future_to_plate = {
            executer.submit(
                lambda path: pd.read_csv(path, compression="gzip", storage_options={"anon": True}),
                cp_feature_source_formatter.format(cell_type=cell_type, plate=plate),
            ): plate
            for plate in plates
        }
        for future in as_completed(future_to_plate):
            per_data = future.result()
            per_data[PERISCOPE_BATCH_COL] = future_to_plate[future]
            per_data_all.append(per_data)

    per_data_all = pd.concat(per_data_all)
    mcols = [
        "Metadata_Foci_Barcode_MatchedTo_GeneCode",
        "Metadata_Foci_Barcode_MatchedTo_Barcode",
        PERISCOPE_BATCH_COL,
    ]
    metadata = per_data_all[mcols]  # type: ignore[call-overload]
    features = per_data_all.drop(mcols, axis=1).dropna(axis=1)  # type: ignore[attr-defined]
    return features, metadata


def load_cpg16_crispr(data_path: str = "data/") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return the JUMP-CP (cpg0016) CRISPR dataset CellProfiler features.
    Find more information about the dataset here: https://www.biorxiv.org/content/10.1101/2023.03.23.534023v2
    We download the metadata first, filter it to CRISPR plates, and load the features for these plates only.
    The metadata is downloaded from here:
        https://zenodo.org/records/7661296/files/jump-cellpainting/metadata-v0.5.0.zip?download=1
    The CellProfiler features corresponding to the appropriate plates are loaded from here:
        https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0016-jump/

    Parameters:
    data_path (str): Path to the directory containing the dataset files.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames named `features` and `metadata`,
        representing the loaded data.
    """
    metadata_source_path = "https://zenodo.org/records/7661296/files/jump-cellpainting/datasets-v0.5.0.zip?download=1"
    plate_file_name = "plate.csv.gz"
    well_file_name = "well.csv.gz"
    crispr_file_name = "crispr.csv.gz"
    plate_file_path = os.path.join(data_path, plate_file_name)
    well_file_path = os.path.join(data_path, well_file_name)
    crispr_file_path = os.path.join(data_path, crispr_file_name)
    if not (os.path.exists(plate_file_path) and os.path.exists(well_file_path) and os.path.exists(crispr_file_path)):
        path_to_zip_file = os.path.join(data_path, "tmp.zip")
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
        cripsr_plates = metadata[
            ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_PlateType"]
        ].drop_duplicates()
        features = []
        with ThreadPoolExecutor(max_workers=10) as executer:
            future_to_plate = {
                executer.submit(
                    lambda path: pd.read_parquet(path, storage_options={"anon": True}),
                    cp_feature_source_formatter.format(**row.to_dict()),
                ): cp_feature_source_formatter.format(**row.to_dict())
                for _, row in cripsr_plates.iterrows()
            }
            for future in as_completed(future_to_plate):
                features.append(future.result())
        pd.concat(features).to_parquet(features_file_path)
    features = pd.read_parquet(features_file_path).dropna(axis=1)
    metadata_cols = metadata.columns
    merged_data = metadata.merge(features, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
    return merged_data.drop(columns=metadata_cols), merged_data[metadata_cols]


def load_gwps(data_type: str, gene_type: str = "all", data_path: str = "data/") -> sc.AnnData:
    """
    Load Replogle et al. 2022 single-cell RNA-seq data for K562 cells.
    Find more information about the dataset here: https://pubmed.ncbi.nlm.nih.gov/35688146/
    Four types of K562 data and downloaded using the links at:
        plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387

    Parameters:
    data_type (str): Type of data to load. Must be either 'raw' or 'normalized'.
        Normalized means Z-normalized by gem_group.
    gene_type (str): Type of genes to load. Must be either 'essential' or 'all'. Defaults to 'all'.
    data_path (str): Path to the directory where the data will be saved.

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

    elif gene_type == "all":
        if data_type == "raw":
            filename = "K562_gwps_raw_singlecell_01.h5ad"
            src = "https://ndownloader.figshare.com/files/35775507"
        elif data_type == "normalized":
            filename = "K562_gwps_normalized_singlecell_01.h5ad"
            src = "https://ndownloader.figshare.com/files/35774440"
        else:
            raise ValueError("data_type must be either raw or normalized")

    else:
        raise ValueError("gene_type must be either essential or all")

    fn = os.path.join(data_path, filename)
    if not os.path.exists(fn):
        wget.download(src, fn)

    adata = sc.read_h5ad(fn)
    adata = adata[:, np.all(~np.isnan(adata.X) & ~np.isinf(adata.X), axis=0)]
    return adata
