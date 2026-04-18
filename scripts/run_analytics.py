from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_paper_assets import discover_run_dirs
from src.tables import infer_model_label, keep_latest_runs, summarize_runs
from src.visuals import (
    centroid_similarity_heatmap,
    confusion_matrix_pair,
    confusion_matrix_plot,
    covariance_spectrum,
    dataset_split_distribution,
    feature_space_pca,
    gda_validation_heatmap,
    margin_by_true_rank,
    per_class_accuracy_delta,
    per_class_performance_plot,
    reliability_curve,
    similarity_margin_by_class,
    true_class_rank_histogram,
    within_between_distance,
    zero_shot_margin_histogram,
    zero_shot_per_class_recoverability,
    zero_shot_prediction_bias,
    zero_shot_topk_curve,
    zero_shot_topk_gap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create standard analytics for a completed run, or the selected first-run zero-shot baseline.")
    parser.add_argument("run_dir", nargs="?", help="Completed run directory. If omitted, the latest selected zero-shot CLIP run is analyzed.")
    parser.add_argument("--input_dir", default="outputs/eurosat", help="Outputs root used when run_dir is omitted.")
    parser.add_argument("--output_dir", default="paper_assets/analytics", help="Analytics root. A run-specific subfolder is created by default.")
    parser.add_argument("--dataset_dir", default="data/eurosat/imagefolder", help="EuroSAT ImageFolder root with train/val/test splits.")
    parser.add_argument("--feature_pack", default=None, help="Optional .npz feature pack for feature-geometry analytics.")
    parser.add_argument("--compare_run", default=None, help="Optional second run for method-validation deltas.")
    parser.add_argument("--include_dataset", action="store_true", help="Also write dataset split distribution analytics.")
    args = parser.parse_args()

    summary = load_summary(Path(args.input_dir))
    run_dir = Path(args.run_dir) if args.run_dir else select_run(summary, method="zeroshot_clip", model_contains="Zero-shot CLIP")
    if not (run_dir / "predictions.csv").exists():
        raise FileNotFoundError(f"Missing predictions.csv in {run_dir}")
    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")

    out_root = Path(args.output_dir)
    out_dir = out_root / safe_name(run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(run_dir / "predictions.csv")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    model_label = infer_label(run_dir)

    summary.to_csv(out_dir / "run_summary.csv", index=False)
    write_metric_summary(out_dir, run_dir, model_label, metrics, predictions)

    if args.include_dataset or not args.run_dir:
        split_counts = collect_split_counts(Path(args.dataset_dir))
        split_counts.to_csv(out_dir / "class_distribution.csv", index=False)
        dataset_split_distribution(split_counts, out_dir)
        zero_shot_topk_gap(summary, out_dir)

    if "top5_predictions" in predictions.columns:
        rank_summary = true_class_rank_histogram(predictions, out_dir, name="true_class_rank_histogram", title=f"True-Class Rank: {model_label}")
        rank_summary.to_csv(out_dir / "true_class_rank_histogram.csv", index=False)
        recoverability = zero_shot_per_class_recoverability(
            predictions,
            out_dir,
            artifact_name="per_class_recoverability",
            title=f"Per-Class Top-k Recoverability: {model_label}",
        )
        recoverability.to_csv(out_dir / "per_class_recoverability.csv", index=False)

    logits_path = select_logits_path(run_dir)
    if logits_path is not None:
        logits = np.load(logits_path)
        topk = zero_shot_topk_curve(predictions, logits, out_dir, artifact_name="topk_curve", title=f"Cumulative Top-k Accuracy: {model_label}")
        topk.to_csv(out_dir / "topk_curve.csv", index=False)
        margins = zero_shot_margin_histogram(predictions, logits, out_dir, artifact_name="margin_histogram", title=f"Margin Distribution: {model_label}")
        margins.to_csv(out_dir / "margin_histogram.csv", index=False)
        if "top5_predictions" in predictions.columns:
            rank_margins = margin_by_true_rank(predictions, logits, out_dir)
            rank_margins.to_csv(out_dir / "margin_by_true_rank.csv", index=False)
        by_class = similarity_margin_by_class(predictions, logits, out_dir, name="margin_by_class", title=f"Margin by Class: {model_label}")
        by_class.to_csv(out_dir / "margin_by_class.csv", index=False)
        calibration = reliability_curve(predictions, logits, out_dir)
        calibration.to_csv(out_dir / "reliability_curve.csv", index=False)

    feature_pack = load_npz_pack(Path(args.feature_pack)) if args.feature_pack else load_feature_packs_for_run(run_dir).get("test")
    if feature_pack is not None:
        support_features, support_targets = support_from_feature_packs(run_dir, load_feature_packs_for_run(run_dir))
        class_names = class_names_from_pack(feature_pack)
        pca_summary = feature_space_pca(
            feature_pack["features"],
            feature_pack["targets"],
            class_names,
            out_dir,
            support_features=support_features,
            support_targets=support_targets,
        )
        pca_summary.to_csv(out_dir / "feature_space_pca.csv", index=False)
        centroid = centroid_similarity_heatmap(feature_pack["features"], feature_pack["targets"], class_names, out_dir)
        centroid.to_csv(out_dir / "class_centroid_similarity.csv")
        distances = within_between_distance(feature_pack["features"], feature_pack["targets"], out_dir)
        distances.to_csv(out_dir / "within_between_distance.csv", index=False)
        spectrum_features = support_features if support_features is not None else feature_pack["features"]
        spectrum_targets = support_targets if support_targets is not None else feature_pack["targets"]
        spectrum = covariance_spectrum(spectrum_features, spectrum_targets, out_dir)
        spectrum.to_csv(out_dir / "covariance_spectrum.csv", index=False)

    bias = zero_shot_prediction_bias(predictions, out_dir, artifact_name="prediction_bias", title=f"Prediction Bias: {model_label}", predicted_label="Predicted share")
    bias.to_csv(out_dir / "prediction_bias.csv", index=False)

    if (run_dir / "confusion_matrix.csv").exists():
        confusion_matrix_plot(run_dir / "confusion_matrix.csv", out_dir, name="confusion_matrix", title=f"Normalized Confusion Matrix: {model_label}")
        remove_png(out_dir / "confusion_matrix.png")
    if (run_dir / "per_class_metrics.csv").exists():
        per_class_performance_plot(run_dir / "per_class_metrics.csv", out_dir, title=f"Per-Class Performance: {model_label}")
        remove_png(out_dir / "per_class_performance_plot.png")

    if args.compare_run:
        write_method_validation(run_dir, Path(args.compare_run), out_dir)

    print(f"Analytics written to {out_dir.resolve()}")


def load_summary(input_dir: Path) -> pd.DataFrame:
    run_dirs = discover_run_dirs(input_dir)
    if not run_dirs:
        return pd.DataFrame()
    summary = summarize_runs(run_dirs)
    if summary.empty:
        return summary
    return keep_latest_runs(summary)


def select_run(summary: pd.DataFrame, method: str, model_contains: str | None = None) -> Path:
    if summary.empty:
        raise FileNotFoundError("No completed runs found. Pass a run_dir explicitly or set --input_dir to an outputs root.")
    candidates = summary[summary["method"] == method].copy()
    if model_contains:
        candidates = candidates[candidates["Model"].str.contains(model_contains, regex=False)]
    if candidates.empty:
        raise RuntimeError(f"No run found for method={method!r}")
    candidates = candidates.sort_values(["Shots", "Top-1 Acc", "run_mtime"], ascending=[False, False, False])
    return Path(candidates.iloc[0]["run_dir"])


def infer_label(run_dir: Path) -> str:
    config_path = run_dir / "config_used.yaml"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return infer_model_label(yaml.safe_load(f))
        except Exception:
            pass
    return run_dir.name


def select_logits_path(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "gda_logits.npy",
        run_dir / "tip_adapter_logits.npy",
        run_dir / "convnext_logits.npy",
        run_dir / "clip_zeroshot_logits.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    all_logits = sorted(run_dir.glob("*logits.npy"))
    return all_logits[0] if all_logits else None


def write_metric_summary(out_dir: Path, run_dir: Path, model_label: str, metrics: dict, predictions: pd.DataFrame) -> None:
    top5_inclusion = None
    rank_2_to_5 = None
    if "top5_predictions" in predictions.columns:
        ranks = predictions.apply(lambda row: rank(row["target_name"], row["top5_predictions"]), axis=1)
        top5_inclusion = float((ranks <= 5).mean())
        rank_2_to_5 = float(((ranks >= 2) & (ranks <= 5)).mean())
    lines = [
        "# Analytics Summary",
        "",
        f"Run: `{run_dir.name}`",
        f"Model: {model_label}",
        "",
        f"Top-1 accuracy: {metrics.get('top1_accuracy', float('nan')):.3f}",
        f"Top-5 accuracy: {metrics.get('top5_accuracy', float('nan')):.3f}",
        f"Macro F1: {metrics.get('macro_f1', float('nan')):.3f}",
    ]
    if top5_inclusion is not None:
        lines.extend([
            f"True class in Top-5: {top5_inclusion:.3f}",
            f"True class ranked 2-5: {rank_2_to_5:.3f}",
        ])
    lines.extend([
        "",
        "Standard analytics:",
        "- rank histogram when Top-5 predictions are available",
        "- cumulative Top-k curve and margins when logits are available",
        "- margin by true-class rank when logits and Top-5 predictions are available",
        "- feature-space geometry when frozen features are cached or provided",
        "- prediction bias",
        "- confusion matrix when available",
        "- per-class performance when available",
    ])
    (out_dir / "analytics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_method_validation(run_dir: Path, compare_run: Path, out_dir: Path) -> None:
    if (run_dir / "confusion_matrix.csv").exists() and (compare_run / "confusion_matrix.csv").exists():
        confusion_matrix_pair(run_dir / "confusion_matrix.csv", compare_run / "confusion_matrix.csv", out_dir)
    if (run_dir / "per_class_metrics.csv").exists() and (compare_run / "per_class_metrics.csv").exists():
        delta = per_class_accuracy_delta(run_dir / "per_class_metrics.csv", compare_run / "per_class_metrics.csv", out_dir)
        delta.to_csv(out_dir / "method_validation_delta_by_class.csv", index=False)
    gda_stats_path = compare_run / "gda_stats.json"
    if gda_stats_path.exists():
        with gda_stats_path.open("r", encoding="utf-8") as f:
            gda_stats = json.load(f)
        search = gda_validation_heatmap(gda_stats, out_dir)
        search.to_csv(out_dir / "method_validation_gda_search.csv", index=False)


def collect_split_counts(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            count = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
            rows.append({"split": split_dir.name, "class_name": class_dir.name, "count": count})
    if not rows:
        raise FileNotFoundError(f"No ImageFolder split/class directories found under {dataset_dir}")
    return pd.DataFrame(rows)


def load_feature_packs_for_run(run_dir: Path) -> dict[str, dict | None]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        return {"train": None, "val": None, "test": None}
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    method = config.get("experiment", {}).get("method")
    if method not in {"zeroshot_clip", "clip_gda", "clip_tip_adapter", "clip_gda_proto"}:
        return {"train": None, "val": None, "test": None}
    exp = config["experiment"]
    clip_cfg = config["clip"]
    data_root = Path(exp["data_root"])
    dataset_name = config.get("dataset", {}).get("name", "eurosat")
    augmentations = list(clip_cfg.get("feature_augmentations", ["identity"]))
    packs = {}
    for split in ("train", "val", "test"):
        path = clip_feature_cache_path(data_root, dataset_name, clip_cfg, split, augmentations)
        packs[split] = load_npz_pack(path) if path.exists() else None
    return packs


def support_from_feature_packs(run_dir: Path, packs: dict[str, dict | None]) -> tuple[np.ndarray | None, np.ndarray | None]:
    support_path = run_dir / "support_indices.json"
    if not support_path.exists() or packs.get("train") is None or packs.get("val") is None:
        return None, None
    with support_path.open("r", encoding="utf-8") as f:
        support = json.load(f)
    train_pack, val_pack = packs["train"], packs["val"]
    pool_features = np.vstack([train_pack["features"], val_pack["features"]])
    pool_targets = np.concatenate([train_pack["targets"], val_pack["targets"]])
    features, targets = [], []
    for item in support.get("indices_by_class", {}).values():
        for idx in item.get("indices", []):
            if int(idx) < len(pool_features):
                features.append(pool_features[int(idx)])
                targets.append(pool_targets[int(idx)])
    if not features:
        return None, None
    return np.asarray(features), np.asarray(targets, dtype=np.int64)


def clip_feature_cache_path(data_root: Path, dataset_name: str, clip_cfg: dict, split: str, feature_augmentations: list[str]) -> Path:
    cache_dir = data_root / "feature_cache" / slugify(dataset_name)
    suffix = augmentations_cache_suffix(feature_augmentations)
    aug_suffix = f"_{suffix}" if suffix else ""
    return cache_dir / f"clip_{clip_cfg['model_name']}_{clip_cfg['pretrained']}_{split}{aug_suffix}.npz".replace("/", "-")


def augmentations_cache_suffix(feature_augmentations: list[str]) -> str:
    if feature_augmentations == ["identity"]:
        return ""
    encoded = json.dumps(feature_augmentations).encode("utf-8")
    return "aug_" + hashlib.sha1(encoded).hexdigest()[:8]


def slugify(value: str) -> str:
    return str(value).lower().replace(" ", "_").replace("/", "-")


def load_npz_pack(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],
        "targets": data["targets"],
        "indices": data["indices"].astype(int).tolist() if "indices" in data else list(range(len(data["targets"]))),
        "paths": data["paths"].astype(str).tolist() if "paths" in data else [],
        "class_names": data["class_names"].astype(str).tolist(),
    }


def class_names_from_pack(pack: dict) -> list[str]:
    labels = pd.DataFrame({"target": pack["targets"], "class_name": pack["class_names"]})
    return labels.drop_duplicates("target").sort_values("target")["class_name"].tolist()


def rank(target_name: str, top5_predictions: str) -> int:
    for idx, pred in enumerate(str(top5_predictions).split("|"), start=1):
        if pred == target_name:
            return idx
    return 6


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def remove_png(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
