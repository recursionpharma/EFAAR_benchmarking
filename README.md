# EFAAR_benchmarking

This library enables computation and retrieval of metrics to benchmark a whole-genome perturbative map created by the pipeline.
Metrics that can be computed using this repo are pairwise gene-gene recall for the Reactome, HuMAP, CORUM, SIGNOR, and StringDB datasets which
are publicly available (see `efaar_benchmarking/benchmark_annotations/LICENSE` for terms of use for each source).

By default, we do not filter on perturbation fingerprint, although filtering is available through the parameters to the `benchmark` function.
We compute the metrics for three different random seeds used to generate empirical null entities.

See our bioRxiv paper for the details: `https://www.biorxiv.org/content/10.1101/2022.12.09.519400v1`

Here are the descriptions for the constants used in the code to configure and control various aspects of the benchmarking process:

`BENCHMARK_DATA_DIR`: The directory path to the benchmark annotations data. It is obtained using the resources module from the importlib package.

`BENCHMARK_SOURCES`: A list of benchmark sources, including "Reactome", "HuMAP", "CORUM", "SIGNOR", and "StringDB".

`PERT_LABEL_COL`: The column name for the gene perturbation labels.

`CONTROL_PERT_LABEL`: The perturbation label value for the control perturbation units.

`PERT_SIG_PVAL_COL`: The column name for the perturbation p-value.

`PERT_SIG_PVAL_THR`: The threshold value for the perturbation p-value.

`RECALL_PERC_THRS`: A list of tuples of two floats between 0 and 1 representing the threshold pair (lower threshold, upper threshold) for calculating recall.

`RANDOM_SEED`: The random seed value used for random number generation for sampling the null distribution.

`RANDOM_COUNT`: The number of runs for benchmarking to compute error in metrics.

`N_NULL_SAMPLES`: The number of null samples used in benchmarking.

`MIN_REQ_ENT_CNT`: The minimum required number of entities for benchmarking.

## Installation

This package is installable via `pip`.

```bash
pip install efaar_benchmarking
```

## Example code:

```from efaar_benchmarking.data_loading import load_replogle
from efaar_benchmarking.efaar import embed_by_scvi, align_by_centering, aggregate_by_mean
from efaar_benchmarking.benchmarking import benchmark
from efaar_benchmarking.plotting import plot_recall

adata = load_replogle("genome_wide", "raw")
metadata = adata.obs
embeddings_scvi = embed_by_scvi(adata)
embeddings_aligned = align_by_centering(embeddings_scvi, metadata)
map_data = aggregate_by_mean(embeddings_aligned, metadata)
metrics = benchmark(map_data,
recall_thr_pairs=[(0.01,0.99),(0.02,0.98),(0.03,0.97),(0.04,0.96),(0.05,0.95),(0.06,0.94),(0.07,0.93),(0.08,0.92),(0.09,0.91),(0.1,0.9)])
plot_recall(metrics["summary"])
```

## References
**Reactome:**

_Gillespie, M., Jassal, B., Stephan, R., Milacic, M., Rothfels, K., Senff-Ribeiro, A., Griss, J., Sevilla, C., Matthews, L., Gong, C., et al. (2022). The reactome pathway knowledgebase 2022. Nucleic Acids Res. 50, D687–D692. 10.1093/nar/gkab1028._

**CORUM:**

_Giurgiu, M., Reinhard, J., Brauner, B., Dunger-Kaltenbach, I., Fobo, G., Frishman, G., Montrone, C., and Ruepp, A. (2019). CORUM: the comprehensive resource of mammalian protein complexes-2019. Nucleic Acids Res. 47, D559–D563. 10.1093/nar/gky973._

**HuMAP:**

_Drew, K., Wallingford, J.B., and Marcotte, E.M. (2021). hu.MAP 2.0: integration of over 15,000 proteomic experiments builds a global compendium of human multiprotein assemblies. Mol. Syst. Biol. 17, e10016. 10.15252/msb.202010016._

**SIGNOR:**

_Licata, L., Lo Surdo, P., Iannuccelli, M., Palma, A., Micarelli, E., Perfetto, L., Peluso, D., Calderone, A., Castagnoli, L., and Cesareni, G. (2019). SIGNOR 2.0, the SIGnaling Network Open Resource 2.0: 2019 update. Nucleic Acids Research. 10.1093/nar/gkz949._

**StringDB:**

_von Mering C, Jensen LJ, Snel B, Hooper SD, Krupp M, Foglierini M, Jouffre N, Huynen MA, Bork P. STRING: known and predicted protein-protein associations, integrated and transferred across organisms. Nucleic Acids Res. 2005 Jan 1;33(Database issue):D433-7. doi: 10.1093/nar/gki005._
