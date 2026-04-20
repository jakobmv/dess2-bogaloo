from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def cosine_similarity_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = l2_normalize(left.astype(np.float32, copy=False))
    right_norm = l2_normalize(right.astype(np.float32, copy=False))
    return np.sum(left_norm * right_norm, axis=1)
