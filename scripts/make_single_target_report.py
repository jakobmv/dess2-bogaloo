from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dess2_bogaloo.single_target import (
    FEATURE_COLUMNS,
    GAS_TURBINE_CITATION,
    SINGLE_TARGET_VARIANT_DESCRIPTIONS,
    TARGET_COLUMNS,
    build_gas_turbine_splits,
    GasTurbinePaths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concise report for the single-target Gas Turbine DESS sweep."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/single_target_gas_turbine"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/single_target_gas_turbine/report.md"),
    )
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


def _collect_runs(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(output_root.glob("**/*.metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        metrics_path = Path(str(metadata["metrics_path"]))
        runtime_path = metadata_path.parent / "runtime_seconds.txt"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        runtime_seconds = None
        if runtime_path.exists():
            runtime_seconds = float(runtime_path.read_text().strip())
        rows.append(
            {
                "variant": metadata["variant"],
                "seed": metrics["seed"],
                "run_name": metrics["name"],
                "test_rmse": metrics["test_rmse"],
                "test_mae": metrics["test_mae"],
                "test_r2": metrics["test_r2"],
                "test_mean_nll": metrics["test_mean_nll"],
                "test_rmse_co": metrics["test_rmse_co"],
                "test_rmse_nox": metrics["test_rmse_nox"],
                "test_r2_co": metrics["test_r2_co"],
                "test_r2_nox": metrics["test_r2_nox"],
                "val_rmse": metrics["val_rmse"],
                "best_epoch": metrics["best_epoch"],
                "train_rows": metrics["train_rows"],
                "val_rows": metrics["val_rows"],
                "test_rows": metrics["test_rows"],
                "runtime_seconds": runtime_seconds,
                "predictions_path": metadata["predictions_path"],
                "metrics_path": metadata["metrics_path"],
                "history_path": metadata["history_path"],
                "checkpoint_path": metadata["checkpoint_path"],
                "description": metadata["variant_description"],
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["variant", "seed"], kind="mergesort").reset_index(drop=True)
    return frame


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    grouped = frame.groupby("variant", sort=False)
    summary = grouped.agg(
        seeds=("seed", "count"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),
        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),
        test_r2_mean=("test_r2", "mean"),
        test_r2_std=("test_r2", "std"),
        test_mean_nll_mean=("test_mean_nll", "mean"),
        test_mean_nll_std=("test_mean_nll", "std"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        runtime_seconds_std=("runtime_seconds", "std"),
        best_epoch_mean=("best_epoch", "mean"),
        test_rmse_co_mean=("test_rmse_co", "mean"),
        test_rmse_nox_mean=("test_rmse_nox", "mean"),
        test_r2_co_mean=("test_r2_co", "mean"),
        test_r2_nox_mean=("test_r2_nox", "mean"),
    ).reset_index()
    summary["description"] = summary["variant"].map(SINGLE_TARGET_VARIANT_DESCRIPTIONS)
    return summary.sort_values("test_rmse_mean", ascending=True, kind="mergesort").reset_index(drop=True)


def _plus_minus(mean: object, std: object, *, digits: int = 4) -> str:
    if mean is None or pd.isna(mean):
        return ""
    if std is None or pd.isna(std):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _dataset_section(data_root: Path) -> list[str]:
    splits = build_gas_turbine_splits(GasTurbinePaths(data_root))
    rows = pd.DataFrame(
        [
            {"split": "train", "rows": int(splits["train"].shape[0])},
            {"split": "validation", "rows": int(splits["val"].shape[0])},
            {"split": "test", "rows": int(splits["test"].shape[0])},
        ]
    )
    rows["rows"] = rows["rows"].map(_format_int)
    return [
        "## Dataset",
        "",
        "Dataset: UCI Gas Turbine CO and NOx Emission Data Set.",
        "",
        f"Citation: {GAS_TURBINE_CITATION}",
        "",
        "Inputs: 9 tabular features.",
        "",
        f"Targets: {', '.join(TARGET_COLUMNS)}.",
        "",
        "Split protocol: 2011-2013 used as the training/cross-validation pool, 2014-2015 used as test; the tail of the train pool is held out as validation while preserving chronology.",
        "",
        _frame_to_markdown(rows),
        "",
    ]


def _aggregate_section(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["## Aggregate Results", "", "_No runs found._", ""]
    export = frame.copy()
    export.to_csv(Path("outputs") / "single_target_gas_turbine_summary.csv", index=False)
    table = pd.DataFrame(
        {
            "variant": frame["variant"],
            "rmse": [
                _plus_minus(mean, std)
                for mean, std in zip(frame["test_rmse_mean"], frame["test_rmse_std"])
            ],
            "mae": [
                _plus_minus(mean, std)
                for mean, std in zip(frame["test_mae_mean"], frame["test_mae_std"])
            ],
            "r2": [
                _plus_minus(mean, std)
                for mean, std in zip(frame["test_r2_mean"], frame["test_r2_std"])
            ],
            "mean_nll": [
                _plus_minus(mean, std)
                for mean, std in zip(frame["test_mean_nll_mean"], frame["test_mean_nll_std"])
            ],
            "rmse_co": frame["test_rmse_co_mean"].map(_format_float),
            "rmse_nox": frame["test_rmse_nox_mean"].map(_format_float),
            "r2_co": frame["test_r2_co_mean"].map(_format_float),
            "r2_nox": frame["test_r2_nox_mean"].map(_format_float),
            "runtime_s": [
                _plus_minus(mean, std)
                for mean, std in zip(frame["runtime_seconds_mean"], frame["runtime_seconds_std"])
            ],
            "seeds": frame["seeds"].map(_format_int),
        }
    )
    best_variant = str(frame.iloc[0]["variant"])
    return [
        "## Aggregate Results",
        "",
        _frame_to_markdown(table),
        "",
        f"Best variant by mean test RMSE: `{best_variant}`.",
        "",
    ]


def _per_run_section(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["## Per-Run Results", "", "_No runs found._", ""]
    table = frame[
        [
            "variant",
            "seed",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_mean_nll",
            "best_epoch",
            "runtime_seconds",
        ]
    ].copy()
    for column in ["test_rmse", "test_mae", "test_r2", "test_mean_nll", "runtime_seconds"]:
        table[column] = table[column].map(_format_float)
    table["seed"] = table["seed"].map(_format_int)
    table["best_epoch"] = table["best_epoch"].map(_format_int)
    return [
        "## Per-Run Results",
        "",
        _frame_to_markdown(table),
        "",
    ]


def _artifact_section(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["## Artifacts", "", "_No runs found._", ""]
    table = frame[
        [
            "variant",
            "seed",
            "predictions_path",
            "metrics_path",
            "history_path",
            "checkpoint_path",
        ]
    ].copy()
    table["seed"] = table["seed"].map(_format_int)
    return [
        "## Artifacts",
        "",
        _frame_to_markdown(table),
        "",
    ]


def main() -> None:
    args = parse_args()
    runs = _collect_runs(args.output_root)
    aggregate = _aggregate(runs)
    lines = ["# Single-Target DESS Report", ""]
    lines.extend(_dataset_section(args.data_root))
    lines.extend(_aggregate_section(aggregate))
    lines.extend(_per_run_section(runs))
    lines.extend(_artifact_section(runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
