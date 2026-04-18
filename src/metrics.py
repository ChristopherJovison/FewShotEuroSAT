from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .utils import topk_accuracy


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    preds = logits.argmax(axis=1)
    top1 = float((preds == targets).mean())
    top5 = topk_accuracy(logits, targets, k=min(5, logits.shape[1]))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=np.arange(len(class_names)))
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        targets, preds, labels=np.arange(len(class_names)), zero_division=0
    )
    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)
    per_acc = np.divide(class_correct, class_totals, out=np.zeros_like(class_correct, dtype=float), where=class_totals > 0)
    per_class = pd.DataFrame({
        "class_id": np.arange(len(class_names)),
        "class_name": class_names,
        "support_count": support,
        "accuracy": per_acc,
        "precision": per_p,
        "recall": per_r,
        "f1": per_f1,
    })
    per_class["difficulty_rank"] = per_class["accuracy"].rank(method="first", ascending=True).astype(int)
    metrics = {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "hardest_classes": per_class.sort_values(["accuracy", "f1"]).head(10)[["class_name", "accuracy", "f1"]].to_dict("records"),
        "easiest_classes": per_class.sort_values(["accuracy", "f1"], ascending=False).head(10)[["class_name", "accuracy", "f1"]].to_dict("records"),
    }
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    return metrics, per_class, cm_df


def predictions_frame(
    logits: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
    paths: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    probs = softmax(logits)
    preds = logits.argmax(axis=1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    rows = []
    for i, pred in enumerate(preds):
        rows.append({
            "index": indices[i],
            "path": paths[i],
            "target": int(targets[i]),
            "target_name": class_names[int(targets[i])],
            "prediction": int(pred),
            "prediction_name": class_names[int(pred)],
            "confidence": float(probs[i, pred]),
            "correct": bool(pred == targets[i]),
            "top5_predictions": "|".join(class_names[int(j)] for j in top5[i]),
            "top5_confidences": "|".join(f"{float(probs[i, j]):.4f}" for j in top5[i]),
        })
    return pd.DataFrame(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def save_metric_outputs(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    run_dir: str | Path,
) -> None:
    import json

    out = Path(run_dir)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_csv(out / "predictions.csv", index=False)
    per_class.to_csv(out / "per_class_metrics.csv", index=False)
    confusion.to_csv(out / "confusion_matrix.csv")
