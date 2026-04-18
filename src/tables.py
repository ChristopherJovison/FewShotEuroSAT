from __future__ import annotations

from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "zeroshot_clip": "Zero-shot CLIP ViT-B/16",
    "clip_gda": "CLIP ViT-B/16 + GDA",
    "clip_gda_proto": "CLIP ViT-B/16 + GDA+Proto",
    "clip_tip_adapter": "CLIP ViT-B/16 + Tip-Adapter",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_tiny_full_finetune": "ConvNeXt-Tiny Full Fine-Tune",
}


def summarize_runs(run_dirs: list[Path]) -> pd.DataFrame:
    import json
    import yaml

    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config_used.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        exp = config["experiment"]
        dataset_cfg = config.get("dataset", {})
        dataset = dataset_cfg.get("name", metrics.get("dataset", "eurosat"))
        model_label = infer_model_label(config)
        rows.append({
            "run_dir": str(run_dir),
            "run_mtime": run_dir.stat().st_mtime,
            "dataset": dataset,
            "method": exp["method"],
            "Model": model_label,
            "Shots": int(exp["shots"]),
            "Seed": int(exp["seed"]),
            "Top-1 Acc": metrics["top1_accuracy"],
            "Top-5 Acc": metrics["top5_accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Runtime": metrics.get("runtime_seconds"),
        })
    return pd.DataFrame(rows)


def keep_latest_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values("run_mtime")
        .drop_duplicates(["dataset", "Model", "Shots", "Seed"], keep="last")
        .sort_values(["dataset", "Shots", "Seed", "method"])
        .reset_index(drop=True)
    )


def export_main_table(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    table = (
        df.sort_values(["Shots", "method"])
        .groupby(["Model", "Shots"], as_index=False)[["Top-1 Acc", "Top-5 Acc", "Macro F1", "Weighted F1"]]
        .mean()
    )
    table = _round_metrics(table)
    out = Path(out_dir)
    _write_csv(table, out / "main_results_table.csv")
    _to_latex(table, out / "main_results_table.tex", bold_by=["Shots"], metric_cols=["Top-1 Acc", "Top-5 Acc", "Macro F1", "Weighted F1"])
    return table


def export_fewshot_table(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    df = df[df["Shots"] > 0].copy()
    pivot = df.pivot_table(index="Shots", columns="Model", values="Top-1 Acc", aggfunc="mean").reset_index()
    pivot = _round_metrics(pivot)
    out = Path(out_dir)
    _write_csv(pivot, out / "fewshot_ablation_table.csv")
    _to_latex(pivot, out / "fewshot_ablation_table.tex", bold_by=["Shots"], metric_cols=[c for c in pivot.columns if c != "Shots"])
    return pivot


def export_per_class_table(run_dir: Path, out_dir: str | Path, stem: str = "per_class_metrics_table") -> pd.DataFrame:
    table = pd.read_csv(run_dir / "per_class_metrics.csv")
    cols = ["class_name", "support_count", "accuracy", "f1", "difficulty_rank"]
    table = table[cols].sort_values("difficulty_rank")
    table = _round_metrics(table)
    out = Path(out_dir)
    _write_csv(table, out / f"{stem}.csv")
    _write_text(table.to_latex(index=False, escape=True, float_format="%.3f"), out / f"{stem}.tex")
    return table


def _round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(3)
    return out


def _to_latex(df: pd.DataFrame, path: Path, bold_by: list[str], metric_cols: list[str]) -> None:
    work = df.copy()
    for col in metric_cols:
        work[col] = work[col].astype(object)
    for _, group in work.groupby(bold_by):
        idx = group.index
        for col in metric_cols:
            original = df.loc[idx, col].astype(float)
            best = original.max()
            mask = original == best
            work.loc[idx[mask], col] = work.loc[idx[mask], col].map(lambda x: f"\\textbf{{{float(x):.3f}}}")
            work.loc[idx[~mask], col] = work.loc[idx[~mask], col].map(lambda x: f"{float(x):.3f}")
    _write_text(work.to_latex(index=False, escape=False), path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"Skipped locked file: {path}")


def _write_text(text: str, path: Path) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        print(f"Skipped locked file: {path}")


def infer_model_label(config: dict) -> str:
    exp = config["experiment"]
    method = exp["method"]
    clip_cfg = config.get("clip", {})
    model_name = str(clip_cfg.get("model_name", ""))
    pretrained = str(clip_cfg.get("pretrained", ""))
    if method == "zeroshot_clip" and pretrained == "remoteclip":
        return "Zero-shot RemoteCLIP ViT-B/32"
    if method == "clip_gda" and pretrained == "remoteclip":
        return "RemoteCLIP ViT-B/32 + GDA"
    if method == "clip_gda" and "quickgelu" in model_name.lower():
        return "CLIP ViT-B/16-QuickGELU + GDA"
    return METHOD_LABELS.get(method, method)
