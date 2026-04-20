from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dess2_bogaloo.eval import evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved reranking run with corrected paper-style NDCG.")
    parser.add_argument("run_csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.run_csv)
    metrics = evaluate_run(frame)
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
