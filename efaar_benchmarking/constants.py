from importlib import resources

BENCHMARK_DATA_DIR = str(resources.files("efaar_benchmarking").joinpath("benchmark_annotations"))
BENCHMARK_SOURCES = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"]
RECALL_PERC_THRS = [(0.05, 0.95), (0.1, 0.9)]
RANDOM_SEED = 42
RANDOM_COUNT = 3
N_NULL_SAMPLES = 5000
MIN_REQ_ENT_CNT = 20
PERT_SIG_PVAL_COL = "gene_pvalue"
PERT_SIG_PVAL_THR = 0.01

REPLOGLE_CONTROL_PERT_LABEL = "non-targeting"
REPLOGLE_PERT_LABEL_COL = "gene"
REPLOGLE_BATCH_COL = "gem_group"

JUMP_CONTROL_PERT_LABEL = "non-targeting"
JUMP_PERT_LABEL_COL = "Metadata_Symbol"
JUMP_BATCH_COL = "Metadata_Plate"
JUMP_BATCH_COL_2 = "Metadata_Batch"

PERISCOPE_CONTROL_PERT_LABEL = "nontargeting"
PERISCOPE_PERT_LABEL_COL = "Metadata_Foci_Barcode_MatchedTo_GeneCode"
PERISCOPE_BATCH_COL = "Metadata_Plate"
PERISCOPE_BATCH_COL_2 = "Metadata_Batch"
