from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dess2_bogaloo.data import DatasetPaths, build_reranking_subset, build_training_subset
from dess2_bogaloo.eval import LABEL_GAINS, add_gain_column, ndcg_for_query


VARIANT_DESCRIPTIONS = {
    "mlp_joint": "One 3-layer MLP jointly predicts the Gaussian mean (mu) and uncertainty (sigma).",
    "frozen_mu_sigma_mlp": "Keeps the frozen query embedding as mu and learns only sigma with a 3-layer MLP.",
    "dual_head_detached_sigma": "Separate mu and sigma heads; sigma is trained against a detached copy of mu.",
    "dual_head_query_concat_sigma": "Separate heads; sigma sees the concatenation of the query embedding and detached mu.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-oriented markdown report for reproduction and DESS runs.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--reproduction-dir", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--dess-dir", type=Path, default=Path("outputs/dess"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report.md"))
    parser.add_argument("--baseline-run", type=str, default="sbert_text")
    parser.add_argument("--top-query-deltas", type=int, default=5)
    return parser.parse_args()


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = frame.columns.tolist()
    rows = frame.values.tolist()
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _format_float(value: object, *, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _find_single(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _read_runtime_seconds(run_dir: Path) -> float | None:
    runtime_path = run_dir / "runtime_seconds.txt"
    if not runtime_path.exists():
        return None
    raw = runtime_path.read_text().strip()
    if not raw:
        return None
    return float(raw)


def _read_first_pass_runtime_seconds(run_dir: Path) -> float | None:
    runtime_path = run_dir / "runtime_seconds_first_pass.txt"
    if not runtime_path.exists():
        return None
    raw = runtime_path.read_text().strip()
    if not raw:
        return None
    return float(raw)


def _collect_run_dirs(dess_dir: Path) -> list[Path]:
    if not dess_dir.exists():
        return []
    run_dirs = {path.parent for path in dess_dir.rglob("*.metadata.json")}
    return sorted(run_dirs)


def _load_dess_runs(dess_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in _collect_run_dirs(dess_dir):
        metadata_path = _find_single(run_dir, "*.metadata.json")
        summary_path = run_dir / "summary.csv"
        history_path = _find_single(run_dir, "*.history.csv")
        if metadata_path is None or not summary_path.exists():
            continue

        metadata = _read_json(metadata_path)
        summary_row = pd.read_csv(summary_path).iloc[0].to_dict()
        history_row: dict[str, object] = {}
        if history_path is not None and history_path.exists():
            history = pd.read_csv(history_path)
            if not history.empty:
                history_row = history.iloc[-1].to_dict()

        config = metadata.get("config", {})
        official_probe = metadata.get("official_multi_target_probe", {})
        rows.append(
            {
                "variant": metadata.get("variant"),
                "description": VARIANT_DESCRIPTIONS.get(str(metadata.get("variant")), ""),
                "run_name": metadata.get("name", summary_row.get("name")),
                "ndcg": summary_row.get("ndcg"),
                "num_queries": summary_row.get("num_queries"),
                "num_judgements": summary_row.get("num_judgements"),
                "train_rows": metadata.get("train_rows"),
                "train_queries": metadata.get("train_queries"),
                "feature_source": metadata.get("feature_source"),
                "loss_impl": metadata.get("loss_impl"),
                "official_probe_ok": official_probe.get("ok"),
                "official_probe_reason": official_probe.get("reason"),
                "batch_size": config.get("batch_size"),
                "eval_batch_size": config.get("eval_batch_size"),
                "epochs": config.get("epochs"),
                "learning_rate": config.get("learning_rate"),
                "weight_decay": config.get("weight_decay"),
                "beta": config.get("beta"),
                "alpha": config.get("alpha"),
                "dropout": config.get("dropout"),
                "hidden_dim": config.get("hidden_dim"),
                "min_gain": config.get("min_gain"),
                "max_train_rows": config.get("max_train_rows"),
                "seed": config.get("seed"),
                "device": config.get("device"),
                "model_name": config.get("model_name"),
                "runtime_seconds": _read_runtime_seconds(run_dir),
                "runtime_seconds_first_pass": _read_first_pass_runtime_seconds(run_dir),
                "final_epoch": history_row.get("epoch"),
                "final_loss": history_row.get("loss"),
                "final_mu_loss": history_row.get("mu_loss"),
                "final_sigma_loss": history_row.get("sigma_loss"),
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "metadata_path": str(metadata_path),
                "history_path": str(history_path) if history_path is not None else "",
                "run_path": metadata.get("run_path", ""),
                "checkpoint_path": metadata.get("checkpoint_path", ""),
                "metrics_path": metadata.get("metrics_path", ""),
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.sort_values("ndcg", ascending=False, kind="mergesort").reset_index(drop=True)


def _load_reproduction_summary(reproduction_dir: Path) -> pd.DataFrame:
    summary_path = reproduction_dir / "summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def _dataset_setup_rows(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = DatasetPaths(data_root)
    train_subset = build_training_subset(paths)
    eval_subset = build_reranking_subset(paths)

    positive_train = train_subset.loc[train_subset["gain"] >= 0.01].copy()
    train_rows = pd.DataFrame(
        [
            {"split": "train", "subset": "all judged rows", "rows": int(train_subset.shape[0]), "queries": int(train_subset["query_id"].nunique())},
            {"split": "train", "subset": "positive rows used for DESS", "rows": int(positive_train.shape[0]), "queries": int(positive_train["query_id"].nunique())},
            {"split": "test", "subset": "reranking candidate list", "rows": int(eval_subset.shape[0]), "queries": int(eval_subset["query_id"].nunique())},
        ]
    )
    gain_rows = pd.DataFrame(
        [
            {"label": label, "gain": gain}
            for label, gain in LABEL_GAINS.items()
        ]
    )
    return train_rows, gain_rows


def _per_query_ndcg(frame: pd.DataFrame) -> pd.DataFrame:
    scored = add_gain_column(frame)
    scored = scored.sort_values(
        ["query_id", "score", "product_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    per_query = (
        scored.groupby("query_id", sort=False)
        .agg(query=("query", "first"), ndcg=("gain", lambda gains: ndcg_for_query(gains.tolist())))
        .reset_index()
    )
    return per_query


def _best_variant_query_deltas(
    *,
    dess_runs: pd.DataFrame,
    reproduction_dir: Path,
    baseline_run: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if dess_runs.empty:
        return None

    best_row = dess_runs.iloc[0]
    best_run_path = Path(str(best_row["run_path"]))
    baseline_path = reproduction_dir / f"{baseline_run}.csv"
    if not best_run_path.exists() or not baseline_path.exists():
        return None

    best_frame = _per_query_ndcg(pd.read_csv(best_run_path))
    baseline_frame = _per_query_ndcg(pd.read_csv(baseline_path))
    merged = baseline_frame.merge(
        best_frame,
        on=["query_id", "query"],
        how="inner",
        suffixes=("_baseline", "_variant"),
    )
    merged["delta_ndcg"] = merged["ndcg_variant"] - merged["ndcg_baseline"]
    merged = merged.sort_values("delta_ndcg", ascending=False, kind="mergesort").reset_index(drop=True)
    improved = merged.head(top_k).copy()
    regressed = merged.tail(top_k).sort_values("delta_ndcg", ascending=True, kind="mergesort").copy()
    return merged, improved, regressed


def _render_reproduction_section(reproduction_dir: Path) -> list[str]:
    summary = _load_reproduction_summary(reproduction_dir)
    if summary.empty:
        return ["## Part 1: SQID Reproduction", "", "_Not available yet._", ""]

    baseline_names = ["random", "esci_baseline", "sbert_text", "clip_text", "clip_image"]
    baselines = summary.loc[summary["name"].isin(baseline_names)].copy()
    baselines["name"] = pd.Categorical(baselines["name"], categories=baseline_names, ordered=True)
    baselines = baselines.sort_values("name")
    baselines["ndcg"] = baselines["ndcg"].map(_format_float)
    baselines["num_judgements"] = baselines["num_judgements"].map(_format_int)
    baselines["num_queries"] = baselines["num_queries"].map(_format_int)

    lines = ["## Part 1: SQID Reproduction", "", _frame_to_markdown(baselines), ""]

    best_fusion_path = reproduction_dir / "best_fusion.json"
    if best_fusion_path.exists():
        best_fusion = _read_json(best_fusion_path)
        lines.extend(
            [
                f"Best fusion baseline: `{best_fusion['name']}` with NDCG `{float(best_fusion['ndcg']):.4f}`.",
                "",
            ]
        )
    return lines


def _render_setup_section(data_root: Path) -> list[str]:
    setup_rows, gain_rows = _dataset_setup_rows(data_root)
    setup_rows = setup_rows.copy()
    setup_rows["rows"] = setup_rows["rows"].map(_format_int)
    setup_rows["queries"] = setup_rows["queries"].map(_format_int)
    gain_rows = gain_rows.copy()
    gain_rows["gain"] = gain_rows["gain"].map(_format_float)

    return [
        "## Experimental Setup",
        "",
        "All DESS runs use the SQID / ESCI reranking task with the per-query judged candidate list, the corrected gains E=1.0, S=0.1, C=0.01, I=0.0, and NDCG as the headline metric.",
        "",
        "Dataset splits used in this repo:",
        "",
        _frame_to_markdown(setup_rows),
        "",
        "Gain mapping:",
        "",
        _frame_to_markdown(gain_rows),
        "",
        "Frozen feature source: SBERT text embeddings from `sentence-transformers/all-MiniLM-L12-v2` over query text and product titles.",
        "",
    ]


def _render_dess_section(
    *,
    dess_runs: pd.DataFrame,
    reproduction_dir: Path,
    baseline_run: str,
) -> list[str]:
    if dess_runs.empty:
        return ["## Part 2: DESS Variants", "", "_Not available yet._", ""]

    reproduction_summary = _load_reproduction_summary(reproduction_dir)
    baseline_lookup = {
        row["name"]: float(row["ndcg"])
        for _, row in reproduction_summary.iterrows()
    }
    sbert_ndcg = baseline_lookup.get(baseline_run)
    random_ndcg = baseline_lookup.get("random")
    esci_ndcg = baseline_lookup.get("esci_baseline")
    best_fusion_path = reproduction_dir / "best_fusion.json"
    best_fusion_ndcg: float | None = None
    best_fusion_name: str | None = None
    if best_fusion_path.exists():
        best_fusion = _read_json(best_fusion_path)
        best_fusion_ndcg = float(best_fusion["ndcg"])
        best_fusion_name = str(best_fusion["name"])

    comparison = dess_runs.copy()
    if sbert_ndcg is not None:
        comparison["delta_vs_sbert"] = comparison["ndcg"] - sbert_ndcg
    if random_ndcg is not None:
        comparison["delta_vs_random"] = comparison["ndcg"] - random_ndcg
    if esci_ndcg is not None:
        comparison["delta_vs_esci_baseline"] = comparison["ndcg"] - esci_ndcg
    if best_fusion_ndcg is not None:
        comparison["delta_vs_best_fusion"] = comparison["ndcg"] - best_fusion_ndcg

    export = comparison.copy()
    export.to_csv(Path("outputs") / "dess_variant_summary.csv", index=False)

    table = comparison[
        [
            "variant",
            "ndcg",
            "delta_vs_sbert",
            "delta_vs_random",
            "delta_vs_esci_baseline",
            "delta_vs_best_fusion",
            "train_rows",
            "train_queries",
            "runtime_seconds",
            "final_loss",
            "final_mu_loss",
            "final_sigma_loss",
        ]
    ].copy()
    for column in ["ndcg", "delta_vs_sbert", "delta_vs_random", "delta_vs_esci_baseline", "delta_vs_best_fusion", "final_loss", "final_mu_loss", "final_sigma_loss"]:
        if column in table.columns:
            table[column] = table[column].map(_format_float)
    for column in ["train_rows", "train_queries"]:
        table[column] = table[column].map(_format_int)
    if "runtime_seconds" in table.columns:
        table["runtime_seconds"] = table["runtime_seconds"].map(_format_float)

    lines = ["## Part 2: DESS Variants", "", _frame_to_markdown(table), ""]

    variant_defs = comparison[["variant", "description"]].copy()
    lines.extend(["Variant definitions:", "", _frame_to_markdown(variant_defs), ""])

    config_table = comparison[
        [
            "variant",
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "beta",
            "alpha",
            "hidden_dim",
            "dropout",
            "device",
            "max_train_rows",
        ]
    ].copy()
    for column in ["learning_rate", "weight_decay", "beta", "alpha", "dropout"]:
        config_table[column] = config_table[column].map(_format_float)
    lines.extend(["Training configuration:", "", _frame_to_markdown(config_table), ""])

    artifacts = comparison[
        [
            "variant",
            "run_path",
            "checkpoint_path",
            "history_path",
            "metrics_path",
            "metadata_path",
        ]
    ].copy()
    lines.extend(["Artifact locations:", "", _frame_to_markdown(artifacts), ""])

    caveats = comparison[
        [
            "variant",
            "loss_impl",
            "official_probe_ok",
            "official_probe_reason",
        ]
    ].copy()
    lines.extend(["Implementation caveats:", "", _frame_to_markdown(caveats), ""])

    best_row = comparison.iloc[0]
    if best_fusion_name is not None and best_fusion_ndcg is not None:
        lines.extend(
            [
                f"Reference strongest reproduction baseline: `{best_fusion_name}` with NDCG `{best_fusion_ndcg:.4f}`.",
                "",
            ]
        )
    warm_runtime_rows = comparison.loc[comparison["runtime_seconds_first_pass"].notna(), ["variant", "runtime_seconds_first_pass"]].copy()
    if not warm_runtime_rows.empty:
        warm_runtime_rows["runtime_seconds_first_pass"] = warm_runtime_rows["runtime_seconds_first_pass"].map(_format_float)
        lines.extend(
            [
                "Initial cache-building pass runtimes captured separately from the warm-cache comparison table:",
                "",
                _frame_to_markdown(warm_runtime_rows),
                "",
            ]
        )
    lines.extend(
        [
            f"Best DESS variant in this sweep: `{best_row['variant']}` with NDCG `{float(best_row['ndcg']):.4f}`.",
            "",
        ]
    )
    return lines


def _render_query_delta_section(
    *,
    dess_runs: pd.DataFrame,
    reproduction_dir: Path,
    baseline_run: str,
    top_k: int,
) -> list[str]:
    delta_payload = _best_variant_query_deltas(
        dess_runs=dess_runs,
        reproduction_dir=reproduction_dir,
        baseline_run=baseline_run,
        top_k=top_k,
    )
    if delta_payload is None:
        return ["## Query-Level Deltas", "", "_Not available yet._", ""]

    merged, improved, regressed = delta_payload
    merged.to_csv(Path("outputs") / "best_variant_vs_sbert_query_deltas.csv", index=False)

    for frame in (improved, regressed):
        frame["ndcg_baseline"] = frame["ndcg_baseline"].map(_format_float)
        frame["ndcg_variant"] = frame["ndcg_variant"].map(_format_float)
        frame["delta_ndcg"] = frame["delta_ndcg"].map(_format_float)

    best_variant = str(dess_runs.iloc[0]["variant"])
    return [
        "## Query-Level Deltas",
        "",
        f"Top query-level changes for the best DESS variant (`{best_variant}`) relative to `{baseline_run}`:",
        "",
        "Largest improvements:",
        "",
        _frame_to_markdown(improved[["query_id", "query", "ndcg_baseline", "ndcg_variant", "delta_ndcg"]]),
        "",
        "Largest regressions:",
        "",
        _frame_to_markdown(regressed[["query_id", "query", "ndcg_baseline", "ndcg_variant", "delta_ndcg"]]),
        "",
    ]


def main() -> None:
    args = parse_args()
    dess_runs = _load_dess_runs(args.dess_dir)

    lines = ["# SQID Reproduction and DESS Variant Report", ""]
    lines.extend(_render_setup_section(args.data_root))
    lines.extend(_render_reproduction_section(args.reproduction_dir))
    lines.extend(
        _render_dess_section(
            dess_runs=dess_runs,
            reproduction_dir=args.reproduction_dir,
            baseline_run=args.baseline_run,
        )
    )
    lines.extend(
        _render_query_delta_section(
            dess_runs=dess_runs,
            reproduction_dir=args.reproduction_dir,
            baseline_run=args.baseline_run,
            top_k=args.top_query_deltas,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
