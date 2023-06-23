from importlib import resources

BENCHMARK_DATA_DIR = resources.files("efaar_benchmarking").joinpath(  # type:ignore[attr-defined]
    "benchmark_annotations"
)
BENCHMARK_SOURCES = ["Reactome", "HuMAP", "CORUM"]
PERT_LABEL_COL = "gene"
PERT_SIG_PVAL_COL = "perturbation_pvalue"
PERT_SIG_PVAL_THR = 0.01
PERT_TYPE_COL = "perturbation_type"
PERT_TYPE = "GENE"
WELL_TYPE_COL = "well_type"
WELL_TYPE = "query_guides"
RECALL_PERC_THR_PAIR = (0.05, 0.95)
RANDOM_SEED = 42
RANDOM_COUNT = 3
N_NULL_SAMPLES = 5000
MIN_REQ_ENT_CNT = 20
