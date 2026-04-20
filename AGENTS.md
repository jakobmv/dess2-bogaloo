# Repo Conventions

Use this repo style as the default unless there is a clear reason not to.

- Use `uv` for environment management, dependency installation, and running commands.
- Keep a small `Makefile` with practical shortcuts for the workflows we actually use.
- Put importable Python code in `src/<project_name>/`.
- Put runnable task entrypoints in `scripts/`.
- Keep README files short, command-oriented, and focused on the main workflow.
- Keep repositories lean. Avoid scaffolding, abstractions, and folders that do not help right now.
- Do not commit large datasets, caches, or generated artifacts. Keep them local and gitignored.
- Prefer simple Python, clear names, and obvious file layout over clever patterns.
- When adding functionality, preserve the existing structure instead of inventing a new one.

# AGENTS.md

## Mission

Build a lean research repo in two parts:

1. Faithfully reproduce the SQID paper’s reranking baselines.
2. Add a 3-layer MLP + DESS reranker as a separate extension.

Do not blur these two stages.

## Core rules

For Part 1, follow the SQID paper setup exactly:

- use the SQID / ESCI Task 1 reranking setup
- use only the per-query judged candidate list
- do not do full-catalog retrieval
- do not redesign the benchmark
- do not add new models before the paper baselines work

For Part 2:

- keep the same reranking task
- keep the same candidate lists
- keep the same evaluation mapping
- add DESS as a new method on top

## Repo conventions

Use this repo style unless there is a clear reason not to:

- Use uv for environment management, dependency installation, and running commands.
- Keep a small Makefile with practical shortcuts.
- Put importable Python code in `src/<project_name>/`.
- Put runnable entrypoints in `scripts/`.
- Keep README files short and command-oriented.
- Keep repositories lean.
- Do not commit large datasets, caches, or generated artifacts.
- Prefer simple Python and clear names over abstractions.

## Part 1: SQID paper reproduction

Implement these baselines:

- Random
- ESCI_Baseline
- SBERT_text
- CLIP_text
- CLIP_image
- text-image fusion variants

Important:
- reproduce the paper’s reranking setup, not a different retrieval task
- score only the products already attached to each query
- use the paper’s corrected gains:
  - E = 1.0
  - S = 0.1
  - C = 0.01
  - I = 0.0

Use the paper’s metric as the headline result:
- NDCG

Extra metrics may be added later, but not before reproduction is working.

## Part 2: DESS extension

Only start after Part 1 is working.

Add a new reranker with:

- frozen features as input
- 3-layer MLP
- DESS final layer

Initial default architecture:

- Linear(d, 1024) -> GELU -> Dropout(0.1)
- Linear(1024, 1024) -> GELU -> Dropout(0.1)
- Linear(1024, 2d)

Split final output into:
- mu
- raw_sigma

Use:
- sigma = softplus(raw_sigma) + 1e-6

Use the official DESS implementation where possible.
Do not rewrite DESS from scratch unless needed for integration.

Keep DESS on the same reranking task.
Do not convert the benchmark into full retrieval or single-target regression in v1.

## File layout

Keep the repo small.

.
├── Makefile
├── README.md
├── pyproject.toml
├── scripts/
│   ├── download_data.py
│   ├── run_reproduction.py
│   ├── evaluate.py
│   ├── train_dess.py
│   └── make_report.py
└── src/<project_name>/
    ├── data.py
    ├── baselines.py
    ├── eval.py
    ├── dess_model.py
    ├── train.py
    └── utils.py

Do not add more structure unless there is a real need.

## Required scripts

### scripts/download_data.py
- fetch or locate SQID resources
- fetch or locate any ESCI metadata needed for faithful evaluation

### scripts/run_reproduction.py
- run all Part 1 baselines
- save scores and rankings

### scripts/evaluate.py
- compute corrected paper-style NDCG
- support both reproduction and DESS runs

### scripts/train_dess.py
- train the 3-layer MLP + DESS reranker
- save checkpoints and metrics

### scripts/make_report.py
- generate a small qualitative report
- clearly separate reproduction results from DESS extension results

## Non-goals for v1

Do not add:

- full-catalog retrieval
- adapters
- LoRA
- encoder fine-tuning
- custom benchmark redesign
- large framework abstractions
- dashboards or apps
- complex training tricks

## Delivery order

Work in this exact order:

1. Load SQID resources
2. Reproduce the exact SQID reranking subset and evaluation
3. Run Random
4. Run ESCI_Baseline
5. Run SBERT_text
6. Run CLIP_text
7. Run CLIP_image
8. Run fusion baselines
9. Verify results are close enough to the paper
10. Add the 3-layer MLP + DESS extension
11. Evaluate DESS with the same protocol
12. Generate final report

Do not start step 10 before step 9 is complete.

## Acceptance criteria

The task is done when:

- SQID reproduction runs on the correct reranking setup
- all paper baselines are implemented
- corrected NDCG evaluation is in place
- reproduced results are in the paper’s ballpark
- DESS runs as a clearly labeled extension
- DESS uses the same benchmark and evaluation protocol
- a concise report is generated
- the repo stays lean and reproducible