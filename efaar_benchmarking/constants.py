from importlib import resources

BENCHMARK_DATA_DIR = str(resources.files("efaar_benchmarking").joinpath("benchmark_annotations"))
BENCHMARK_SOURCES = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"]
RECALL_PERC_THRS = [(0.05, 0.95), (0.1, 0.9)]
RANDOM_SEED = 42
N_NULL_SAMPLES = 5000
MIN_REQ_ENT_CNT = 20
