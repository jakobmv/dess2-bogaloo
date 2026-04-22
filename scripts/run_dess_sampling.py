from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from dess2_bogaloo.data import DatasetPaths
from dess2_bogaloo.dess_sampling import DessSamplingConfig, run_dess_sampling_reranker
from dess2_bogaloo.dess_model import VARIANT_MODEL_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DESS reranking by sampling from the saved mu/sigma distributions."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("outputs/dess"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dess_sampling"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_MODEL_TYPES),
        choices=sorted(VARIANT_MODEL_TYPES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    return parser.parse_args()


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda")
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda"


def find_checkpoint(checkpoints_root: Path, variant: str) -> Path:
    matches = sorted((checkpoints_root / variant).glob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for variant {variant} under {checkpoints_root / variant}")
    return matches[0]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows: list[dict[str, object]] = []
    for variant in args.variants:
        checkpoint_path = find_checkpoint(args.checkpoints_root, variant)
        for seed in args.seeds:
            output_dir = args.output_root / variant / f"seed_{seed}"
            started = time.time()
            metrics = run_dess_sampling_reranker(
                paths=DatasetPaths(args.data_root),
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                config=DessSamplingConfig(
                    seed=seed,
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                ),
            )
            runtime_seconds = time.time() - started
            (output_dir / "runtime_seconds.txt").write_text(f"{runtime_seconds:.3f}\n")
            rows.append({**metrics, "runtime_seconds": runtime_seconds})
            print(f"{variant} seed={seed} complete -> {output_dir}")

    summary = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    aggregate = (
        summary.groupby("variant", sort=False)
        .agg(
            seeds=("seed", "count"),
            ndcg_mean=("ndcg", "mean"),
            ndcg_std=("ndcg", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            runtime_seconds_std=("runtime_seconds", "std"),
        )
        .reset_index()
        .sort_values("ndcg_mean", ascending=False, kind="mergesort")
    )
    aggregate.to_csv(args.output_root / "aggregate_summary.csv", index=False)


if __name__ == "__main__":
    main()
