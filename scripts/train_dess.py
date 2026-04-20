from __future__ import annotations

import argparse
from pathlib import Path

from dess2_bogaloo.train import ensure_reproduction_verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage gate for the DESS extension.")
    parser.add_argument(
        "--reproduction-summary",
        type=Path,
        default=Path("outputs/reproduction/summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_reproduction_verified(args.reproduction_summary)
    raise NotImplementedError(
        "DESS training is intentionally gated until reproduction has been verified in this repo."
    )


if __name__ == "__main__":
    main()
