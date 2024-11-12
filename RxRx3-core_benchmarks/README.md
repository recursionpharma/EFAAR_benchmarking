## RxRx3-core Gene-Gene and Gene-compound relationship benchmark

In `rxrx3_core_benchmarks_openphenom.ipynb` we leverage a specialized benchmark to run the gene-gene [EFAAR benchmarks](../) and also to measure compound activity against a gene. This is based on our RxRx3-core dataset hosted on [Hugging Face](https://huggingface.co/datasets/recursionpharma/rxrx3-core).

This benchmark evaluates the zero-shot prediction of compound-gene activity using cosine similarities between model embeddings. Specifically, for each compound, we assess whether the cosine similarities correctly rank the compound's known target genes higher than a randomly sampled set of other genes from the ground truth dataset.

To achieve this, we compute the cosine similarity between each compound and gene across all available concentrations and take the maximum similarity score for each pair. This approach captures the strongest potential interaction regardless of concentration, even if negatives come from different concentrations than positives.

We then treat the absolute value of the cosine similarity as a confidence measure—similar to a classifier's probability score—and compute the AUC (Area Under the ROC Curve) and average precision for each compound. The final results report the median AUC and average precision across all compounds, compared against a random baseline.

## Accessing RxRx3-core

Loading the RxRx3-core image dataset. A notebook for computing embeddings on this dataset with OpenPhenom-S/16 is provided [here](https://huggingface.co/recursionpharma/OpenPhenom/blob/main/RxRx3-core_inference.ipynb).
```
from datasets import load_dataset
rxrx3_core = load_dataset("recursionpharma/rxrx3-core")
```
Loading OpenPhenom-S/16 embeddings and metadata for RxRx3-core
```
from huggingface_hub import hf_hub_download
import pandas as pd

file_path_metadata = hf_hub_download("recursionpharma/rxrx3-core", filename="metadata_rxrx3_core.csv",repo_type="dataset")
file_path_embs = hf_hub_download("recursionpharma/rxrx3-core", filename="OpenPhenom_rxrx3_core_embeddings.parquet",repo_type="dataset")

open_phenom_embeddings = pd.read_parquet(file_path_embs)
rxrx3_core_metadata = pd.read_csv(file_path_metadata)
```
Benchmarking code for this dataset is provided in `rxrx3_core_benchmarks_openphenom.ipynb`.
