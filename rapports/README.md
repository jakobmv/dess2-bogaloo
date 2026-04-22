# Result Package

This folder is a compact shareable subset of `outputs/`.

It keeps the files that are useful for reviewing results:

- experiment reports
- aggregate summary tables
- per-run metrics
- per-run metadata
- training histories
- runtime logs

It intentionally omits the large artifacts:

- model checkpoints
- embedding caches
- full reranking CSVs
- full prediction CSVs
- smoke runs

## What To Open First

- `sqid_reranking/report.md`
  Main multi-target SQID reproduction + DESS variant report.
- `sqid_sampling/aggregate_summary.csv`
  Aggregate results for the sampling-based DESS reranker.
- `single_target_gas_turbine/report.md`
  Report for the single-target DESS benchmark.

## Folder Guide

- `sqid_reranking/`
  Paper-style reranking report, reproduction summaries, DESS variant summaries, and compact per-variant run artifacts.
- `sqid_sampling/`
  Sampling-reranker aggregate table, per-seed summary table, and compact per-run artifacts.
- `single_target_gas_turbine/`
  Single-target report, aggregate table, and compact per-run artifacts.

## Notes

- Some copied reports still mention original paths under `outputs/`; those heavy artifacts were intentionally not included here.
- If someone later needs full ranked lists or checkpoints, they should be taken from the original `outputs/` directory, not from this package.
