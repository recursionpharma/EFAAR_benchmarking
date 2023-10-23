import os
import wget
import scanpy as sc


def load_replogle(gene_type, data_type, data_path="data/"):
    """
    Load Replogle et al. 2020 single-cell RNA-seq data for K562 cells.

    Parameters:
    gene_type (str): Type of genes to load. Must be either 'essential' or 'genome_wide'.
    data_type (str): Type of data to load. Must be either 'raw' or 'normalized'.
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

    return sc.read_h5ad(data_path + filename)
