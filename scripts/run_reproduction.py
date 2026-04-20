from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dess2_bogaloo.baselines import (
    clip_image_baseline,
    clip_text_baseline,
    combine_runs,
    random_baseline,
    sbert_text_baseline,
    score_esci_baseline,
    train_esci_baseline_model,
)
from dess2_bogaloo.data import DatasetPaths, build_reranking_subset
from dess2_bogaloo.eval import evaluate_run
from dess2_bogaloo.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQID Part 1 reranking baselines.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["random"],
        choices=[
            "random",
            "esci_baseline",
            "sbert_text",
            "clip_text",
            "clip_image",
            "fusion",
        ],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esci-model-dir", type=Path, default=None)
    parser.add_argument("--esci-random-state", type=int, default=42)
    parser.add_argument("--esci-n-dev-queries", type=int, default=400)
    parser.add_argument("--esci-train-batch-size", type=int, default=32)
    parser.add_argument("--esci-max-train-rows", type=int, default=None)
    return parser.parse_args()


def save_run(output_dir: Path, name: str, frame: pd.DataFrame, metrics: dict[str, float]) -> None:
    ensure_dir(output_dir)
    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.metrics.json"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, metrics)


def write_summary(output_dir: Path) -> None:
    rows: list[dict[str, float | int | str]] = []
    for metrics_path in sorted(output_dir.glob("*.metrics.json")):
        metrics = pd.read_json(metrics_path, typ="series").to_dict()
        rows.append({"name": metrics_path.name.removesuffix(".metrics.json"), **metrics})
    if rows:
        pd.DataFrame(rows).sort_values("ndcg", ascending=False).to_csv(
            output_dir / "summary.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()
    paths = DatasetPaths(args.data_dir)
    subset = build_reranking_subset(paths)
    ensure_dir(args.output_dir)

    completed: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}

    if "random" in args.baselines:
        result = random_baseline(subset, seed=args.seed)
        metrics = evaluate_run(result.frame)
        save_run(args.output_dir, result.name, result.frame, metrics)
        completed[result.name] = (result.frame, metrics)

    if "sbert_text" in args.baselines:
        result = sbert_text_baseline(subset, cache_dir=args.output_dir / "cache")
        metrics = evaluate_run(result.frame)
        save_run(args.output_dir, result.name, result.frame, metrics)
        completed[result.name] = (result.frame, metrics)

    if "clip_text" in args.baselines:
        result = clip_text_baseline(subset, paths)
        metrics = evaluate_run(result.frame)
        save_run(args.output_dir, result.name, result.frame, metrics)
        completed[result.name] = (result.frame, metrics)

    if "clip_image" in args.baselines:
        result = clip_image_baseline(subset, paths)
        metrics = evaluate_run(result.frame)
        save_run(args.output_dir, result.name, result.frame, metrics)
        completed[result.name] = (result.frame, metrics)

    if "esci_baseline" in args.baselines:
        model_dir = train_esci_baseline_model(
            paths=paths,
            model_dir=args.esci_model_dir or (args.output_dir / "models" / "esci_baseline"),
            random_state=args.esci_random_state,
            n_dev_queries=args.esci_n_dev_queries,
            train_batch_size=args.esci_train_batch_size,
            max_train_rows=args.esci_max_train_rows,
        )
        result = score_esci_baseline(subset, model_dir=model_dir)
        metrics = evaluate_run(result.frame)
        save_run(args.output_dir, result.name, result.frame, metrics)
        completed[result.name] = (result.frame, metrics)

    if "fusion" in args.baselines:
        class _Container:
            def __init__(self, name: str, frame: pd.DataFrame) -> None:
                self.name = name
                self.frame = frame

        required_runs: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}
        for baseline_name in ("sbert_text", "clip_text", "clip_image"):
            run = completed.get(baseline_name)
            if run is None:
                if baseline_name == "sbert_text":
                    result = sbert_text_baseline(subset, cache_dir=args.output_dir / "cache")
                elif baseline_name == "clip_text":
                    result = clip_text_baseline(subset, paths)
                else:
                    result = clip_image_baseline(subset, paths)
                run = (result.frame, evaluate_run(result.frame))
                completed[baseline_name] = run
                save_run(args.output_dir, baseline_name, result.frame, run[1])
            required_runs[baseline_name] = run

        sweep_rows: list[dict[str, float | str]] = []
        best_name = ""
        best_score = -1.0
        pairs = [
            ("sbert_text", "clip_image"),
            ("clip_text", "clip_image"),
        ]
        for left_name, right_name in pairs:
            for method in ("score", "rank"):
                for alpha in (0.1, 0.25, 0.5, 0.75, 0.9):
                    result = combine_runs(
                        _Container(left_name, required_runs[left_name][0]),
                        _Container(right_name, required_runs[right_name][0]),
                        alpha=alpha,
                        method=method,
                    )
                    metrics = evaluate_run(result.frame)
                    save_run(args.output_dir, result.name, result.frame, metrics)
                    sweep_rows.append(
                        {
                            "name": result.name,
                            "left": left_name,
                            "right": right_name,
                            "alpha": alpha,
                            "method": method,
                            **metrics,
                        }
                    )
                    if metrics["ndcg"] > best_score:
                        best_name = result.name
                        best_score = metrics["ndcg"]
        pd.DataFrame(sweep_rows).sort_values("ndcg", ascending=False).to_csv(
            args.output_dir / "fusion_summary.csv",
            index=False,
        )
        write_json(args.output_dir / "best_fusion.json", {"name": best_name, "ndcg": best_score})

    write_summary(args.output_dir)


if __name__ == "__main__":
    main()
