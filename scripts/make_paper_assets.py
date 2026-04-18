from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tables import export_fewshot_table, export_main_table, export_per_class_table, keep_latest_runs, summarize_runs
from src.visuals import accuracy_vs_shots, comparison_barplot, confusion_matrix_plot, per_class_performance_plot, qualitative_grid, training_free_clip_family_barplot


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run outputs into paper-ready assets.")
    parser.add_argument("--input_dir", default="outputs/latest", help="A run directory or an outputs root containing run directories.")
    parser.add_argument("--output_dir", default="paper_assets")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_run_dirs(input_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No completed run directories found under {input_dir}")
    summary = summarize_runs(run_dirs)
    if summary.empty:
        raise RuntimeError("No metrics/config pairs found to aggregate.")
    summary = keep_latest_runs(summary)
    write_csv(summary, out_dir / "all_runs_summary.csv")
    main_table = export_main_table(summary, out_dir)
    fewshot_table = export_fewshot_table(summary, out_dir)
    comparison_barplot(summary, out_dir)
    training_free_clip_family_barplot(summary, out_dir)
    accuracy_vs_shots(summary, out_dir)

    proposed_run = select_proposed_run(summary)
    export_per_class_table(proposed_run, out_dir, stem="remoteclip_gda_per_class_metrics_table")
    confusion_matrix_plot(
        proposed_run / "confusion_matrix.csv",
        out_dir,
        name="remoteclip_gda_confusion_matrix",
        title="Normalized Confusion Matrix: RemoteCLIP + GDA",
    )
    per_class_performance_plot(
        proposed_run / "per_class_metrics.csv",
        out_dir,
        title="Per-Class Performance: RemoteCLIP + GDA",
        name="remoteclip_gda_per_class_performance_plot",
    )
    build_qualitative(summary, out_dir)
    write_summary(out_dir, summary, main_table, fewshot_table, proposed_run)
    print(f"Paper assets written to {out_dir.resolve()}")


def discover_run_dirs(input_dir: Path) -> list[Path]:
    if (input_dir / "metrics.json").exists():
        root = input_dir.parent if input_dir.name == "latest" else input_dir.parent
    else:
        root = input_dir
    return sorted([p for p in root.rglob("*") if p.is_dir() and (p / "metrics.json").exists() and p.name != "latest"])


def select_proposed_run(summary: pd.DataFrame) -> Path:
    dataset = summary.sort_values(["Shots", "Top-1 Acc"], ascending=[False, False]).iloc[0]["dataset"]
    proposed = summary[
        (summary["method"] == "clip_gda")
        & (summary["dataset"] == dataset)
        & (summary["Model"].str.contains("RemoteCLIP", regex=False))
    ].sort_values(["Shots", "Top-1 Acc"], ascending=[False, False])
    if proposed.empty:
        raise RuntimeError("No RemoteCLIP + GDA run found; needed for proposed-method assets.")
    return Path(proposed.iloc[0]["run_dir"])


def build_qualitative(summary: pd.DataFrame, out_dir: Path) -> None:
    proposed = summary[
        (summary["method"] == "clip_gda")
        & summary["Model"].str.contains("RemoteCLIP", regex=False)
    ].sort_values(["Shots", "Top-1 Acc"], ascending=[False, False])
    if proposed.empty:
        return
    proposed_row = proposed.iloc[0]
    dataset = proposed_row["dataset"]
    shots, seed = int(proposed_row["Shots"]), int(proposed_row["Seed"])
    selected = summary[(summary["dataset"] == dataset) & (summary["Shots"] == shots) & (summary["Seed"] == seed)]

    zero = selected[
        (selected["method"] == "zeroshot_clip")
        & selected["Model"].str.contains("RemoteCLIP", regex=False)
    ].sort_values("Top-1 Acc", ascending=False)
    if zero.empty:
        zero = selected[selected["method"] == "zeroshot_clip"].sort_values("Top-1 Acc", ascending=False)
    convnext = selected[selected["method"] == "convnext_tiny"].sort_values("Top-1 Acc", ascending=False)
    if zero.empty or convnext.empty:
        return

    base = pd.read_csv(Path(zero.iloc[0]["run_dir"]) / "predictions.csv")[["index", "path", "target_name", "prediction_name", "confidence", "correct"]].copy()
    base = base.rename(columns={"prediction_name": "zero_prediction_name", "confidence": "zero_confidence", "correct": "zero_correct"})
    gda = pd.read_csv(Path(proposed_row["run_dir"]) / "predictions.csv")[["index", "prediction_name", "confidence", "correct"]].rename(columns={"prediction_name": "gda_prediction_name", "confidence": "gda_confidence", "correct": "gda_correct"})
    conv = pd.read_csv(Path(convnext.iloc[0]["run_dir"]) / "predictions.csv")[["index", "prediction_name", "confidence", "correct"]].rename(columns={"prediction_name": "convnext_prediction_name", "confidence": "convnext_confidence", "correct": "convnext_correct"})
    merged = base.merge(gda, on="index").merge(conv, on="index")
    qualitative_grid(merged, out_dir)


def write_summary(out_dir: Path, summary: pd.DataFrame, main_table: pd.DataFrame, fewshot_table: pd.DataFrame, proposed_run: Path) -> None:
    best = summary.sort_values("Top-1 Acc", ascending=False).iloc[0]
    dataset_names = ", ".join(str(name) for name in sorted(summary["dataset"].unique()))
    lines = [
        "# Experiment Summary",
        "",
        f"Dataset: {dataset_names}.",
        "",
        f"Aggregated runs: {len(summary)}",
        f"Best Top-1 run: {best['Model']} at {int(best['Shots'])}-shot, Top-1={best['Top-1 Acc']:.3f}.",
        f"Proposed-method run: `{proposed_run.name}` (RemoteCLIP + GDA).",
        "",
        "Primary paper files:",
        "- `main_results_table.csv` / `.tex`",
        "- `fewshot_ablation_table.csv` / `.tex`",
        "- `accuracy_comparison_barplot.pdf`",
        "- `summary_clip_family_barplot.pdf`",
        "- `accuracy_vs_shots_plot.pdf`",
        "- `remoteclip_gda_confusion_matrix.pdf`",
        "- `remoteclip_gda_per_class_performance_plot.pdf`",
        "- `remoteclip_gda_per_class_metrics_table.csv` / `.tex`",
        "- `qualitative_examples_grid.pdf` when all three prediction files are available",
    ]
    write_text("\n".join(lines), out_dir / "experiment_summary.md")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"Skipped locked file: {path}")


def write_text(text: str, path: Path) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        print(f"Skipped locked file: {path}")


if __name__ == "__main__":
    main()
