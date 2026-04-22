from __future__ import annotations

import argparse
from pathlib import Path

from dess2_bogaloo.data import DatasetPaths
from dess2_bogaloo.dess_model import VARIANT_MODEL_TYPES
from dess2_bogaloo.train import (
    DESS_FEATURE_SOURCES,
    DessTrainConfig,
    ensure_reproduction_verified,
    train_and_evaluate_dess,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen-feature + 3-layer adapter + DESS reranker."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dess"),
    )
    parser.add_argument(
        "--reproduction-summary",
        type=Path,
        default=Path("outputs/reproduction/summary.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=50000,
        help="Maximum positive training rows to use. Set to 0 or a negative value to use all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--feature-source",
        type=str,
        default="sbert_text",
        choices=sorted(DESS_FEATURE_SOURCES),
        help="Frozen feature source to use for query/product embeddings.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="mlp_joint",
        choices=sorted(VARIANT_MODEL_TYPES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_reproduction_verified(args.reproduction_summary)
    max_train_rows = args.max_train_rows if args.max_train_rows and args.max_train_rows > 0 else None
    config = DessTrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        alpha=args.alpha,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        min_gain=args.min_gain,
        max_train_rows=max_train_rows,
        seed=args.seed,
        device=args.device,
        variant=args.variant,
        feature_source=args.feature_source,
    )
    train_and_evaluate_dess(
        paths=DatasetPaths(args.data_root),
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
