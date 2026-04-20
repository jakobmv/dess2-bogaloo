from __future__ import annotations

from pathlib import Path


def ensure_reproduction_verified(summary_path: Path) -> None:
    if not summary_path.exists():
        raise FileNotFoundError(
            "Reproduction summary not found. Per AGENTS.md, DESS work starts only after reproduction is verified."
        )
