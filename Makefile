.PHONY: sync test download-data reproduce-random reproduce evaluate report train-dess

sync:
	uv sync

test:
	uv run python -m unittest

download-data:
	uv run python scripts/download_data.py

reproduce-random:
	uv run python scripts/run_reproduction.py --baselines random

reproduce:
	uv run python scripts/run_reproduction.py --baselines random sbert_text clip_text clip_image

evaluate:
	uv run python scripts/evaluate.py outputs/reproduction/random.csv

report:
	uv run python scripts/make_report.py

train-dess:
	uv run python scripts/train_dess.py
