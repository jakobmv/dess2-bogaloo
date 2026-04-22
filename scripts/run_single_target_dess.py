from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dess2_bogaloo.single_target import (
    SINGLE_TARGET_VARIANT_DESCRIPTIONS,
    SingleTargetConfig,
    train_single_target_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-target DESS variants on the UCI Gas Turbine CO/NOx regression task."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/single_target_gas_turbine"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(SINGLE_TARGET_VARIANT_DESCRIPTIONS),
        choices=sorted(SINGLE_TARGET_VARIANT_DESCRIPTIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    for variant in args.variants:
        for seed in args.seeds:
            output_dir = args.output_root / variant / f"seed_{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            config = SingleTargetConfig(
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                beta=args.beta,
                alpha=args.alpha,
                dropout=args.dropout,
                hidden_dim=args.hidden_dim,
                val_fraction=args.val_fraction,
                seed=seed,
                device=device,
                variant=variant,
            )
            started = time.time()
            train_single_target_variant(
                data_root=args.data_root,
                output_dir=output_dir,
                config=config,
            )
            (output_dir / "runtime_seconds.txt").write_text(f"{time.time() - started:.3f}\n")
            print(f"{variant} seed={seed} complete -> {output_dir}")


if __name__ == "__main__":
    main()
