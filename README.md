# dess2-bogaloo

Lean research repo for:

1. reproducing the SQID paper's reranking setup and baselines
2. adding a separate 3-layer MLP + DESS reranker only after reproduction is verified

## Main Workflow

```bash
make sync
make test
make download-data
make reproduce-random
```

This repo is pinned to Python `3.12` because the baseline stack depends on PyTorch, and the official PyTorch install docs currently recommend Python `3.9-3.12` on Linux.

Run more reproduction baselines:

```bash
uv run python scripts/run_reproduction.py --baselines random sbert_text clip_text clip_image
```

Evaluate a saved run:

```bash
uv run python scripts/evaluate.py outputs/reproduction/random.csv
```

Build the short report:

```bash
make report
```

`scripts/train_dess.py` is intentionally gated until the reproduction summary exists.
