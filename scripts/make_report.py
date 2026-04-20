from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a concise markdown report.")
    parser.add_argument("--reproduction-dir", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--dess-dir", type=Path, default=Path("outputs/dess"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report.md"))
    return parser.parse_args()


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    rows = frame.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_table(path: Path, title: str) -> list[str]:
    if not path.exists():
        return [f"## {title}", "", "_Not available yet._", ""]
    frame = pd.read_csv(path)
    return [
        f"## {title}",
        "",
        _frame_to_markdown(frame),
        "",
    ]


def _render_reproduction_section(reproduction_dir: Path) -> list[str]:
    summary_path = reproduction_dir / "summary.csv"
    if not summary_path.exists():
        return ["## Part 1: SQID Reproduction", "", "_Not available yet._", ""]

    frame = pd.read_csv(summary_path)
    baseline_names = ["random", "esci_baseline", "sbert_text", "clip_text", "clip_image"]
    baselines = frame.loc[frame["name"].isin(baseline_names)].copy()
    baselines["name"] = pd.Categorical(baselines["name"], categories=baseline_names, ordered=True)
    baselines = baselines.sort_values("name")

    lines = ["## Part 1: SQID Reproduction", ""]
    if not baselines.empty:
        lines.extend(
            [
                "Core baselines:",
                "",
                _frame_to_markdown(baselines.astype({"num_judgements": int, "num_queries": int})),
                "",
            ]
        )

    best_fusion_path = reproduction_dir / "best_fusion.json"
    fusion_summary_path = reproduction_dir / "fusion_summary.csv"
    if best_fusion_path.exists() and fusion_summary_path.exists():
        best_fusion = json.loads(best_fusion_path.read_text())
        fusion_frame = pd.read_csv(fusion_summary_path).head(5)
        lines.extend(
            [
                f"Best fusion: `{best_fusion['name']}` with NDCG `{best_fusion['ndcg']:.4f}`",
                "",
                "Top fusion rows:",
                "",
                _frame_to_markdown(fusion_frame.astype({"num_judgements": int, "num_queries": int})),
                "",
            ]
        )

    return lines


def main() -> None:
    args = parse_args()
    lines = ["# SQID Reproduction and DESS Extension Report", ""]
    lines.extend(_render_reproduction_section(args.reproduction_dir))
    lines.extend(_render_table(args.dess_dir / "summary.csv", "Part 2: DESS Extension"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
