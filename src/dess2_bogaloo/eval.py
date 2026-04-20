from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


LABEL_GAINS = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0.0,
}


def gain_for_label(label: str) -> float:
    return LABEL_GAINS[label]


def add_gain_column(
    frame: pd.DataFrame,
    *,
    label_col: str = "esci_label",
    gain_col: str = "gain",
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[gain_col] = enriched[label_col].map(LABEL_GAINS).astype(float)
    return enriched


def dcg(gains: Iterable[float]) -> float:
    values = np.asarray(list(gains), dtype=np.float64)
    if values.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, values.size + 2, dtype=np.float64))
    return float(np.sum(values * discounts))


def ndcg_for_query(gains: Iterable[float]) -> float:
    values = np.asarray(list(gains), dtype=np.float64)
    if values.size == 0:
        return 0.0
    ideal = np.sort(values)[::-1]
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg(values) / ideal_dcg


def evaluate_run(
    frame: pd.DataFrame,
    *,
    query_col: str = "query_id",
    score_col: str = "score",
    label_col: str = "esci_label",
) -> dict[str, float]:
    scored = add_gain_column(frame, label_col=label_col)
    scored = scored.sort_values(
        [query_col, score_col, "product_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    per_query = (
        scored.groupby(query_col, sort=False)["gain"]
        .apply(ndcg_for_query)
        .astype(float)
    )
    return {
        "ndcg": float(per_query.mean()),
        "num_queries": int(per_query.shape[0]),
        "num_judgements": int(scored.shape[0]),
    }
