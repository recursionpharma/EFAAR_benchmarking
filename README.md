# EFAAR_benchmarking

This library contains functions to build and benchmark a whole-genome perturbative map.

See our bioRxiv paper for details: `https://www.biorxiv.org/content/10.1101/2022.12.09.519400v1`

Here are the descriptions for the constants used in the code to configure and control various aspects of the map building and benchmarking process:

`BENCHMARK_DATA_DIR`: The directory path to the benchmark annotations data. It is obtained using the resources module from the importlib package.

`BENCHMARK_SOURCES`: A list of benchmark sources, including "Reactome", "HuMAP", "CORUM", "SIGNOR", and "StringDB".

`RECALL_PERC_THRS`: A list of tuples of two floats between 0 and 1 representing the threshold pair (lower threshold, upper threshold) for calculating recall.

`RANDOM_SEED`: The random seed value used for random number generation for sampling the null distribution.

`RANDOM_COUNT`: The number of runs for benchmarking to compute error in metrics.

`N_NULL_SAMPLES`: The number of null samples used in benchmarking.

`MIN_REQ_ENT_CNT`: The minimum required number of entities for benchmarking.

Besides above parameters, for each map we build, we utilize distinct constants to indicate the metadata columns for the perturbation, control, and batch information.

## Installation:

This package is installable via `pip`.

```bash
pip install efaar_benchmarking
```

## Example code:

Please see `notebooks/map_building_benchmarking.ipynb`.

## Contribution guidance:

Please see `CONTRIBUTING.md`.

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
