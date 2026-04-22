# SQID + DESS v1

## Goal

Build a lean research codebase with two stages:

1. Faithfully reproduce the SQID paper’s reranking setup and baselines.
2. Add a new 3-layer MLP + DESS reranker on top of the same pipeline.

Part 1 is reproduction.
Part 2 is extension.

## References

- SQID paper: https://huggingface.co/papers/2405.15190
- SQID repo: https://github.com/Crossing-Minds/shopping-queries-image-dataset
- SQID dataset: https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset
- ESCI repo: https://github.com/amazon-science/esci-data
- DESS repo: https://github.com/neddi-as/DESS-Dimensional-PDFs-for-Embedding-Space-Sampling

## Benchmark

Use the SQID paper setup exactly for reproduction:

- ESCI Task 1 reranking
- `small_version=1`
- `split=test`
- `product_locale=us`
- rank only the per-query judged candidate list

This is not full-catalog retrieval.

## Part 1: faithful SQID reproduction

Reproduce these baselines:

- Random
- ESCI_Baseline
- SBERT_text
- CLIP_text
- CLIP_image
- text-image fusion variants

Use the same evaluation style as the paper:

- headline metric: NDCG
- corrected gains:
  - E = 1.0
  - S = 0.1
  - C = 0.01
  - I = 0.0

The goal is not to match every decimal exactly.
The goal is to get close enough that the implementation is clearly correct.

## Part 2: DESS extension

After Part 1 works, add a new reranker:

- frozen encoder features as input
- 3-layer MLP
- DESS final layer

This must run on the same per-query candidate lists and use the same evaluation protocol as Part 1.

Do not change the task into full-catalog retrieval.
Do not change the task into one-query-one-target regression in v1.

## Repo style

- use uv
- keep code in `src/<project_name>/`
- keep runnable scripts in `scripts/`
- keep a small Makefile
- keep datasets, caches, embeddings, and outputs local and gitignored
- prefer simple Python and obvious structure

## v1 done when

- SQID data/resources load locally
- all SQID paper baselines run on the correct reranking setup
- corrected NDCG evaluation is implemented
- reproduced scores are in the right ballpark
- a 3-layer MLP + DESS reranker runs on the same benchmark
- DESS results are reported separately from reproduction results
- a small qualitative report is produced