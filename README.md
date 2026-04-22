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

Train the first DESS extension pass on frozen `SBERT_text` features:

```bash
make train-dess
```

`scripts/train_dess.py` still checks that the reproduction summary exists first, but it now trains a 3-layer adapter on top of the frozen SBERT-text baseline and evaluates it on the same reranking protocol.

Run the separate single-target Gas Turbine benchmark sweep:

```bash
make single-target
make single-target-report
```

This experiment uses the UCI Gas Turbine CO/NOx dataset as a simpler vector regression task for the single-target form of DESS.
