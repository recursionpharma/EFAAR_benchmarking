# EFAAR_benchmarking

This library enables computation and retrieval of metrics to benchmark a whole-genome perturbative map created by the pipeline.
Metrics that can be computed using this repo are pairwise gene-gene recall for the reactome, corum, and humap datasets which 
are publicly available.

By default, we do not filter on perturbation fingerprint, although filtering is available through the parameters to the `benchmark` function.
We compute the metrics for three different random seeds used to generate empirical null entities.

See our bioRxiv paper for the details: `https://www.biorxiv.org/content/10.1101/2022.12.09.519400v1`

Here are the descriptions for the constants used in the code:
`BENCHMARK_DATA_DIR`: The directory path to the benchmark annotations data. It is obtained using the resources module from the importlib package.
`BENCHMARK_SOURCES`: A list of benchmark sources, including "Reactome", "HuMAP", and "CORUM".
`PERT_LABEL_COL`: The column name for the gene perturbation labels.
`PERT_SIG_PVAL_COL`: The column name for the perturbation p-value.
`PERT_SIG_PVAL_THR`: The threshold value for the perturbation p-value.
`PERT_TYPE_COL`: The column name for the perturbation type.
`PERT_TYPE`: The specific perturbation type, which is "GENE" by default.
`WELL_TYPE_COL`: The column name for the well type.
`WELL_TYPE`: The specific well type, which is "query_guides".
`RECALL_PERC_THR_PAIR`: A tuple representing the threshold pair (lower threshold, upper threshold) for calculating recall percentages.
`RANDOM_SEED`: The random seed value used for random number generation.
`RANDOM_COUNT`: The number of runs for benchmarking.
`N_NULL_SAMPLES`: The number of null samples used in benchmarking.
`MIN_REQ_ENT_CNT`: The minimum required number of entities for benchmarking.
These constants are used to configure and control various aspects of the benchmarking process.

## Installation

### PIP

This package is installable via `pip`.

```bash
pip install efaar_benchmarking
```