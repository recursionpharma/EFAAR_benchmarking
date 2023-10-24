from importlib import resources

BENCHMARK_DATA_DIR = resources.files("efaar_benchmarking").joinpath(  # type:ignore[attr-defined]
    "benchmark_annotations"
)
BENCHMARK_SOURCES = ["Reactome", "HuMAP", "CORUM", "SIGNOR", "StringDB"]
PERT_LABEL_COL = "perturbation"
CONTROL_PERT_LABEL = "non-targeting"
PERT_SIG_PVAL_COL = "perturbation_pvalue"
PERT_SIG_PVAL_THR = 0.01
RECALL_PERC_THRS = [(0.05, 0.95), (0.1, 0.9)]
RANDOM_SEED = 42
RANDOM_COUNT = 3
N_NULL_SAMPLES = 5000
MIN_REQ_ENT_CNT = 20
