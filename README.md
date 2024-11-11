# EFAAR_benchmarking

This library contains functions to build and benchmark a whole-genome perturbative map.

See our bioRxiv paper for details: `https://www.biorxiv.org/content/10.1101/2022.12.09.519400v1`

Here are the descriptions for the constants used in the code to configure and control various aspects of the map building and benchmarking process:

`BENCHMARK_DATA_DIR`: The directory path to the benchmark annotations data. It is obtained using the resources module from the importlib package.

`BENCHMARK_SOURCES`: A list of benchmark sources, including "Reactome", "HuMAP", "CORUM", "SIGNOR", and "StringDB".

`RECALL_PERC_THRS`: A list of tuples of two floats between 0 and 1 representing the threshold pair (lower threshold, upper threshold) for calculating recall.

`RANDOM_SEED`: The random seed value used for random number generation for sampling the null distribution.

`N_NULL_SAMPLES`: The number of null samples used in benchmarking.

`MIN_REQ_ENT_CNT`: The minimum required number of entities for benchmarking.

Besides above parameters, for each map we build, we utilize distinct constants to indicate the metadata columns for the perturbation, control, and batch information.

## Installation:

This package is installable via `pip`.

```bash
pip install efaar_benchmarking
```

## Usage guidance:

First, run `notebooks/map_building_benchmarking.ipynb` for GWPS, cpg0016, and cpg0021 individually. This process will build each of these maps and report the perturbation signal and biological relationship benchmarks. Afterwards, run `notebooks/map_evaluation_comparison.ipynb` to explore the constructed maps using the methods presented in our paper. In order for the latter notebook to work, make sure to set the `save_results` parameter to True in the former notebook.

`notebooks/rxrx3_benchmarking.ipynb` contains an example of extracting gene-gene relationships recall and compound-gene average precision scores for the public Rxrx3 dataset.

We've uploaded the 128-dimensional PCA-TVN maps we constructed for GWPS, cpg0016, and cpg0021 to the `notebooks/data` directory. So, for convenience, one can run `notebooks/map_evaluation_comparison.ipynb` directly on these uploaded map files if they wish to explore the maps further without running `notebooks/map_building_benchmarking.ipynb`. See `notebooks/data/LICENSE` for terms of use for each dataset.

RxRx3 embeddings are available as separate parquet files per plate in the embeddings.tar file, downloadable from https://rxrx3.rxrx.ai/downloads. Note that in this data, all but 733 genes are anonymized.

You have to install GitHub LFS (Large File Storage) to properly utilize the files under `notebooks/data` and `efaar_benchmarking/expression_data` in your local clone of the repository.

## Contribution guidance:

See `CONTRIBUTING.md` for code contribution guidance.

If you want to add any new annotation sources, make sure to follow the same format as in the `benchmark_annotations` folder. Your annotations file needs to be a file with a `.txt` extension. It is expected to be a comma-separated two column data frame where column names are `entity1` and `entity2`.

## References for the relationship annotation sources:

See `efaar_benchmarking/benchmark_annotations/LICENSE` for terms of use for each source.

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

**ChEMBL:**

_Zdrazil B, Felix E, Hunter F, Manners EJ, Blackshaw J, Corbett S, de Veij M, Ioannidis H, Lopez DM, Mosquera JF, Magarinos MP, Bosc N, Arcila R, Kizilören T, Gaulton A, Bento AP, Adasme MF, Monecke P, Landrum GA, Leach AR. The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Res. 2024 Jan 5;52(D1):D1180-D1192._

**BindingDB:**

_Gilson,M.K., Liu,T., Baitaluk,M., Nicola,G., Hwang, L. and Chong,J. BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology Nucleic Acids Research 44:D1045-D1053 (2015)._

**Guide to Pharmacology:**

_Harding SD, Armstrong JF, Faccenda E, Southan C, Alexander SPH, Davenport AP, Spedding M, Davies JA. (2023) The IUPHAR/BPS Guide to PHARMACOLOGY in 2024. Nucl. Acids Res. 2024; 52(D1):D1438-D1449. doi:10.1093/nar/gkad944. [Full text]. PMID: 37897341._


## Gene-compound relationship benchmark
In `notebooks/rxrx3_core_benchmarks_openphenom.ipynb` we leverage a specialized benchmark to measure compound activity against a gene.

This benchmark evaluates the zero-shot prediction of compound-gene activity using cosine similarities between model embeddings. Specifically, for each compound, we assess whether the cosine similarities correctly rank the compound's known target genes higher than a randomly sampled set of other genes from the ground truth dataset.

To achieve this, we compute the cosine similarity between each compound and gene across all available concentrations and take the maximum similarity score for each pair. This approach captures the strongest potential interaction regardless of concentration, even if negatives come from different concentrations than positives.

We then treat the absolute value of the cosine similarity as a confidence measure—similar to a classifier's probability score—and compute the AUC (Area Under the ROC Curve) and average precision for each compound. The final results report the median AUC and average precision across all compounds, compared against a random baseline.
