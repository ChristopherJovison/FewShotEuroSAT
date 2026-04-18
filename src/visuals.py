from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import patheffects as pe
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("default")


MODEL_COLORS = {
    "Zero-shot CLIP ViT-B/16": "#3B6FB6",
    "CLIP ViT-B/16 + GDA": "#C43B4D",
    "CLIP ViT-B/16-QuickGELU + GDA": "#B22234",
    "RemoteCLIP ViT-B/32 + GDA": "#D97706",
    "CLIP ViT-B/16 + Tip-Adapter": "#8A63D2",
    "ConvNeXt-Tiny Full Fine-Tune": "#2E8B57",
    "ConvNeXt-Tiny": "#2F8F6B",
    "Zero-shot RemoteCLIP ViT-B/32": "#5B7CBA",
}

MODEL_SHORT_NAMES = {
    "Zero-shot CLIP ViT-B/16": "Zero-shot CLIP",
    "CLIP ViT-B/16 + GDA": "CLIP + GDA",
    "CLIP ViT-B/16-QuickGELU + GDA": "QuickGELU + GDA",
    "RemoteCLIP ViT-B/32 + GDA": "RemoteCLIP + GDA",
    "CLIP ViT-B/16 + Tip-Adapter": "CLIP + Tip-Adapter",
    "ConvNeXt-Tiny Full Fine-Tune": "ConvNeXt FT",
    "ConvNeXt-Tiny": "ConvNeXt-Tiny",
    "Zero-shot RemoteCLIP ViT-B/32": "Zero-shot RemoteCLIP",
}

PAPER_BG = "#FBFAF7"
AXIS_TEXT = "#202124"
MUTED_TEXT = "#6B7280"
GRID_COLOR = "#D7D2C8"


def save_all_formats(fig, out_dir: str | Path, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    except PermissionError:
        print(f"Skipped locked file: {out / f'{name}.pdf'}")
    plt.close(fig)


def _finalize_diagnostic(fig, out_dir: str | Path, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out / f"diagnostic_{name}.pdf", bbox_inches="tight")
    except PermissionError:
        print(f"Skipped locked file: {out / f'diagnostic_{name}.pdf'}")
    plt.close(fig)


def _apply_paper_style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(PAPER_BG)
    ax.figure.set_facecolor(PAPER_BG)
    ax.tick_params(colors=AXIS_TEXT, labelsize=10)
    ax.xaxis.label.set_color(AXIS_TEXT)
    ax.yaxis.label.set_color(AXIS_TEXT)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#A8A29A")
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)


def _dataset_title(summary: pd.DataFrame) -> str:
    if "dataset" not in summary.columns or summary["dataset"].nunique() != 1:
        return "Few-Shot Benchmark"
    return str(summary["dataset"].iloc[0]).replace("_", " ").replace("-", " ").title()


def _focus_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in summary.columns or summary["dataset"].nunique() <= 1:
        return summary.copy()
    candidates = (
        summary.groupby("dataset")
        .agg(max_shots=("Shots", "max"), runs=("run_dir", "count"), best_acc=("Top-1 Acc", "max"))
        .sort_values(["max_shots", "runs", "best_acc"], ascending=False)
    )
    return summary[summary["dataset"] == candidates.index[0]].copy()


def _short_model_name(name: str) -> str:
    return MODEL_SHORT_NAMES.get(name, name)


def _spread_positions(values: list[float], lower: float, upper: float, min_gap: float = 0.035) -> list[float]:
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * len(values)
    last = lower - min_gap
    for original_idx, value in indexed:
        y = min(max(float(value), lower), upper)
        y = max(y, last + min_gap)
        adjusted[original_idx] = y
        last = y
    overflow = adjusted[indexed[-1][0]] - upper
    if overflow > 0:
        for original_idx, _ in reversed(indexed):
            adjusted[original_idx] -= overflow
        for pos in range(len(indexed) - 2, -1, -1):
            idx = indexed[pos][0]
            next_idx = indexed[pos + 1][0]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx] - min_gap)
        underflow = lower - adjusted[indexed[0][0]]
        if underflow > 0:
            for original_idx, _ in indexed:
                adjusted[original_idx] += underflow
    return adjusted


def dataset_split_distribution(split_counts: pd.DataFrame, out_dir: str | Path) -> None:
    df = split_counts.copy()
    split_order = [s for s in ["train", "val", "test"] if s in set(df["split"])]
    class_order = sorted(df["class_name"].unique())
    pivot = (
        df.pivot_table(index="class_name", columns="split", values="count", aggfunc="sum", fill_value=0)
        .reindex(class_order)
        .reindex(columns=split_order)
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    _apply_paper_style(ax, grid_axis="y")
    bottom = np.zeros(len(pivot))
    colors = {"train": "#3B6FB6", "val": "#D97706", "test": "#2F8F6B"}
    for split in split_order:
        values = pivot[split].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=split, color=colors.get(split, "#777777"), edgecolor="#222222", linewidth=0.3)
        bottom += values
    ax.set_ylabel("Images")
    ax.set_xlabel("")
    ax.set_title("EuroSAT Split Distribution", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _finalize_diagnostic(fig, out_dir, "class_distribution")


def zero_shot_topk_gap(summary: pd.DataFrame, out_dir: str | Path) -> None:
    labels = ["Zero-shot CLIP ViT-B/16", "Zero-shot RemoteCLIP ViT-B/32"]
    df = summary[summary["Model"].isin(labels)].copy()
    if df.empty:
        return
    df = df.sort_values(["Model", "Shots"]).drop_duplicates("Model", keep="last")
    df["Label"] = df["Model"].map(_short_model_name)
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    _apply_paper_style(ax, grid_axis="y")
    ax.bar(x - width / 2, df["Top-1 Acc"], width=width, color="#C43B4D", label="Top-1", edgecolor="#222222", linewidth=0.4)
    ax.bar(x + width / 2, df["Top-5 Acc"], width=width, color="#3B6FB6", label="Top-5", edgecolor="#222222", linewidth=0.4)
    for xpos, value in zip(x - width / 2, df["Top-1 Acc"]):
        ax.text(xpos, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color=AXIS_TEXT)
    for xpos, value in zip(x + width / 2, df["Top-5 Acc"]):
        ax.text(xpos, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color=AXIS_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Zero-Shot Top-1 vs. Top-5 Gap", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _finalize_diagnostic(fig, out_dir, "zero_shot_top1_top5")


def true_class_rank_histogram(predictions: pd.DataFrame, out_dir: str | Path, name: str, title: str) -> pd.DataFrame:
    ranks = predictions.apply(_true_rank_from_top5, axis=1)
    counts = ranks.value_counts().reindex([1, 2, 3, 4, 5, 6], fill_value=0)
    labels = ["1", "2", "3", "4", "5", ">5"]
    shares = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    _apply_paper_style(ax, grid_axis="y")
    bars = ax.bar(labels, shares.to_numpy(), color=["#2F8F6B"] + ["#D97706"] * 4 + ["#C43B4D"], edgecolor="#222222", linewidth=0.4)
    for bar, value in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=AXIS_TEXT)
    ax.set_ylim(0, min(1.0, max(0.12, float(shares.max()) + 0.12)))
    ax.set_xlabel("Rank of true class in zero-shot Top-5")
    ax.set_ylabel("Share of test images")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    _finalize_diagnostic(fig, out_dir, name)
    return pd.DataFrame({"rank": labels, "count": counts.to_numpy(), "share": shares.to_numpy()})


def zero_shot_topk_curve(
    predictions: pd.DataFrame,
    logits: np.ndarray,
    out_dir: str | Path,
    artifact_name: str = "zero_shot_topk_curve",
    title: str = "Zero-Shot Cumulative Top-k Accuracy",
    subtitle: str = "The curve tests whether CLIP's logits contain the right class before adaptation",
) -> pd.DataFrame:
    y = predictions["target"].to_numpy(dtype=int)
    order = np.argsort(-logits, axis=1)
    rows = []
    for k in range(1, logits.shape[1] + 1):
        rows.append({"k": k, "accuracy": float(np.any(order[:, :k] == y[:, None], axis=1).mean())})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    _apply_paper_style(ax, grid_axis="y")
    ax.plot(df["k"], df["accuracy"], marker="o", markersize=5.5, linewidth=2.4, color="#3B6FB6")
    ax.scatter([1, 5], df.loc[df["k"].isin([1, 5]), "accuracy"], s=90, color="#C43B4D", edgecolor=PAPER_BG, linewidth=1.2, zorder=3)
    for _, row in df[df["k"].isin([1, 5])].iterrows():
        ax.text(row["k"], row["accuracy"] + 0.025, f"{row['accuracy']:.3f}", ha="center", va="bottom", fontsize=9, color=AXIS_TEXT)
    ax.set_xticks(df["k"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Allowed rank k")
    ax.set_ylabel("True class included in Top-k")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    _finalize_diagnostic(fig, out_dir, artifact_name)
    return df


def zero_shot_margin_histogram(
    predictions: pd.DataFrame,
    logits: np.ndarray,
    out_dir: str | Path,
    artifact_name: str = "zero_shot_margin_histogram",
    title: str = "Zero-Shot Margin Distribution",
    subtitle: str = "Near-zero negative margins indicate boundary/ranking errors rather than missing signal",
) -> pd.DataFrame:
    y = predictions["target"].to_numpy(dtype=int)
    true_scores = logits[np.arange(len(logits)), y]
    masked = logits.copy()
    masked[np.arange(len(masked)), y] = -np.inf
    best_other = masked.max(axis=1)
    margins = true_scores - best_other
    df = predictions[["target_name", "correct"]].copy()
    df["margin"] = margins
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _apply_paper_style(ax, grid_axis="y")
    ax.hist(margins, bins=42, color="#3B6FB6", alpha=0.78, edgecolor=PAPER_BG, linewidth=0.5)
    ax.axvline(0, color="#111111", linewidth=1.2, linestyle="--")
    q25, q50, q75 = np.quantile(margins, [0.25, 0.5, 0.75])
    ax.axvline(q50, color="#C43B4D", linewidth=1.6)
    ax.text(q50, ax.get_ylim()[1] * 0.92, f"median {q50:.3f}", ha="left", va="top", fontsize=9, color="#C43B4D")
    ax.set_xlabel("True-class logit minus strongest competing logit")
    ax.set_ylabel("Test images")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    _finalize_diagnostic(fig, out_dir, artifact_name)
    return pd.DataFrame({
        "statistic": ["mean", "median", "q25", "q75", "share_positive", "share_near_boundary"],
        "value": [
            float(np.mean(margins)),
            float(q50),
            float(q25),
            float(q75),
            float((margins > 0).mean()),
            float(((margins <= 0) & (margins >= -0.01)).mean()),
        ],
    })


def margin_by_true_rank(predictions: pd.DataFrame, logits: np.ndarray, out_dir: str | Path) -> pd.DataFrame:
    y = predictions["target"].to_numpy(dtype=int)
    true_scores = logits[np.arange(len(logits)), y]
    masked = logits.copy()
    masked[np.arange(len(masked)), y] = -np.inf
    best_other = masked.max(axis=1)
    margins = true_scores - best_other
    ranks = predictions.apply(_true_rank_from_top5, axis=1).to_numpy()
    labels = np.array(["1", "2", "3", "4", "5", ">5"])
    rank_labels = labels[np.clip(ranks, 1, 6) - 1]
    df = pd.DataFrame({"true_rank": rank_labels, "margin": margins})
    order = ["1", "2", "3", "4", "5", ">5"]
    data = [df[df["true_rank"] == rank]["margin"].to_numpy() for rank in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    _apply_paper_style(ax, grid_axis="y")
    parts = ax.violinplot(data, positions=np.arange(len(order)), showmeans=False, showmedians=True, widths=0.78)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor("#2F8F6B" if idx == 0 else "#D97706" if idx < 5 else "#C43B4D")
        body.set_edgecolor("#222222")
        body.set_alpha(0.62)
    parts["cmedians"].set_color("#111111")
    parts["cmedians"].set_linewidth(1.4)
    ax.axhline(0, color="#111111", linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Rank assigned to the true class")
    ax.set_ylabel("True-class logit minus strongest competing logit")
    ax.set_title("Margin by True-Class Rank", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    counts = df["true_rank"].value_counts().reindex(order, fill_value=0)
    ymin, ymax = ax.get_ylim()
    for idx, rank in enumerate(order):
        ax.text(idx, ymin + (ymax - ymin) * 0.04, f"n={counts[rank]}", ha="center", va="bottom", fontsize=8.5, color=MUTED_TEXT)
    _finalize_diagnostic(fig, out_dir, "margin_by_true_rank")
    return df.groupby("true_rank")["margin"].agg(["count", "mean", "median", "std"]).reindex(order).reset_index()


def similarity_margin_by_class(predictions: pd.DataFrame, logits: np.ndarray, out_dir: str | Path, name: str, title: str) -> pd.DataFrame:
    y = predictions["target"].to_numpy(dtype=int)
    true_scores = logits[np.arange(len(logits)), y]
    masked = logits.copy()
    masked[np.arange(len(masked)), y] = -np.inf
    best_other = masked.max(axis=1)
    margins = true_scores - best_other
    df = predictions[["target_name"]].copy()
    df["margin"] = margins
    ordered = df.groupby("target_name")["margin"].median().sort_values().index.tolist()
    data = [df[df["target_name"] == cls]["margin"].to_numpy() for cls in ordered]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    _apply_paper_style(ax, grid_axis="x")
    parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#3B6FB6")
        body.set_edgecolor("#222222")
        body.set_alpha(0.55)
    parts["cmedians"].set_color("#C43B4D")
    parts["cmedians"].set_linewidth(1.5)
    ax.axvline(0, color="#111111", linewidth=1.0, linestyle="--")
    ax.set_yticks(np.arange(1, len(ordered) + 1))
    ax.set_yticklabels(ordered, fontsize=8.5)
    ax.set_xlabel("True-class logit minus strongest competing logit")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    _finalize_diagnostic(fig, out_dir, name)
    return df.groupby("target_name")["margin"].agg(["mean", "median", "std", "count"]).reset_index()


def zero_shot_prediction_bias(
    predictions: pd.DataFrame,
    out_dir: str | Path,
    artifact_name: str = "zero_shot_prediction_bias",
    title: str = "Zero-Shot Prediction Bias",
    predicted_label: str = "Predicted share",
    subtitle: str = "A skewed predicted label distribution signals a miscalibrated decision boundary",
) -> pd.DataFrame:
    true_dist = predictions["target_name"].value_counts(normalize=True).rename("true_share")
    pred_dist = predictions["prediction_name"].value_counts(normalize=True).rename("predicted_share")
    df = pd.concat([true_dist, pred_dist], axis=1).fillna(0.0)
    df["overprediction"] = df["predicted_share"] - df["true_share"]
    df = df.sort_values("overprediction")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _apply_paper_style(ax, grid_axis="x")
    ax.barh(y - 0.18, df["true_share"], height=0.34, label="True test share", color="#2F8F6B", edgecolor="#222222", linewidth=0.3)
    ax.barh(y + 0.18, df["predicted_share"], height=0.34, label=predicted_label, color="#C43B4D", edgecolor="#222222", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8.5)
    ax.set_xlabel("Share of test set")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _finalize_diagnostic(fig, out_dir, artifact_name)
    return df.reset_index(names="class_name")


def zero_shot_per_class_recoverability(
    predictions: pd.DataFrame,
    out_dir: str | Path,
    artifact_name: str = "zero_shot_per_class_recoverability",
    title: str = "Zero-Shot Per-Class Recoverability",
    subtitle: str = "Large gaps mean the correct class is usually present but not selected",
) -> pd.DataFrame:
    df = predictions[["target_name", "correct"]].copy()
    df["in_top5"] = predictions.apply(lambda row: _true_rank_from_top5(row) <= 5, axis=1)
    summary = df.groupby("target_name").agg(top1_accuracy=("correct", "mean"), top5_inclusion=("in_top5", "mean")).reset_index()
    summary["recoverability_gap"] = summary["top5_inclusion"] - summary["top1_accuracy"]
    summary = summary.sort_values("recoverability_gap")
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _apply_paper_style(ax, grid_axis="x")
    ax.hlines(y, summary["top1_accuracy"], summary["top5_inclusion"], color="#B8B1A7", linewidth=2.0)
    ax.scatter(summary["top1_accuracy"], y, s=46, color="#C43B4D", edgecolor=PAPER_BG, linewidth=0.8, label="Top-1")
    ax.scatter(summary["top5_inclusion"], y, s=46, color="#3B6FB6", edgecolor=PAPER_BG, linewidth=0.8, label="Top-5 inclusion")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["target_name"], fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Share of class examples")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _finalize_diagnostic(fig, out_dir, artifact_name)
    return summary


def feature_space_pca(
    features: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    out_dir: str | Path,
    support_features: np.ndarray | None = None,
    support_targets: np.ndarray | None = None,
    max_points_per_class: int = 250,
) -> pd.DataFrame:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(42)
    selected = []
    for cls in sorted(np.unique(targets)):
        idx = np.where(targets == cls)[0]
        if len(idx) > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        selected.append(idx)
    sample_idx = np.concatenate(selected)
    fit_features = features[sample_idx]
    if support_features is not None:
        fit_features = np.vstack([fit_features, np.asarray(support_features, dtype=np.float64)])
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(fit_features)
    test_proj = projected[: len(sample_idx)]
    support_proj = projected[len(sample_idx):] if support_features is not None else None

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    _apply_paper_style(ax, grid_axis="both")
    cmap = plt.get_cmap("tab10")
    for cls in sorted(np.unique(targets)):
        mask = targets[sample_idx] == cls
        color = cmap(int(cls) % 10)
        ax.scatter(test_proj[mask, 0], test_proj[mask, 1], s=12, alpha=0.38, color=color, label=class_names[int(cls)], linewidths=0)
        cls_points = test_proj[mask]
        if len(cls_points) >= 3:
            _add_cov_ellipse(ax, cls_points, color=color)
        center = cls_points.mean(axis=0)
        ax.scatter(center[0], center[1], marker="*", s=150, color=color, edgecolor="#111111", linewidth=0.7, zorder=4)
    if support_proj is not None and support_targets is not None:
        ax.scatter(support_proj[:, 0], support_proj[:, 1], s=42, facecolors="none", edgecolors="#111111", linewidth=0.9, label="Support", zorder=5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title("Frozen Feature Space PCA", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="best")
    _finalize_diagnostic(fig, out_dir, "feature_space_pca")
    return pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })


def centroid_similarity_heatmap(features: np.ndarray, targets: np.ndarray, class_names: list[str], out_dir: str | Path) -> pd.DataFrame:
    features = _l2_normalize(np.asarray(features, dtype=np.float64))
    targets = np.asarray(targets, dtype=np.int64)
    classes = sorted(np.unique(targets))
    centroids = []
    labels = []
    for cls in classes:
        center = features[targets == cls].mean(axis=0)
        centroids.append(center)
        labels.append(class_names[int(cls)])
    centroids = _l2_normalize(np.asarray(centroids))
    sim = centroids @ centroids.T
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Class Centroid Cosine Similarity", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", color=AXIS_TEXT)
    _finalize_diagnostic(fig, out_dir, "class_centroid_similarity")
    return pd.DataFrame(sim, index=labels, columns=labels)


def within_between_distance(features: np.ndarray, targets: np.ndarray, out_dir: str | Path, max_pairs: int = 60000) -> pd.DataFrame:
    features = _l2_normalize(np.asarray(features, dtype=np.float64))
    targets = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(42)
    n = len(features)
    first = rng.integers(0, n, size=max_pairs)
    second = rng.integers(0, n, size=max_pairs)
    keep = first != second
    first, second = first[keep], second[keep]
    distances = 1.0 - np.sum(features[first] * features[second], axis=1)
    same = targets[first] == targets[second]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    _apply_paper_style(ax, grid_axis="y")
    bins = np.linspace(float(distances.min()), float(distances.max()), 44)
    ax.hist(distances[same], bins=bins, density=True, alpha=0.72, color="#2F8F6B", label="Within class", edgecolor=PAPER_BG, linewidth=0.4)
    ax.hist(distances[~same], bins=bins, density=True, alpha=0.62, color="#C43B4D", label="Between class", edgecolor=PAPER_BG, linewidth=0.4)
    ax.set_xlabel("Cosine distance in frozen feature space")
    ax.set_ylabel("Density")
    ax.set_title("Within-Class vs. Between-Class Feature Distance", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False)
    _finalize_diagnostic(fig, out_dir, "within_between_distance")
    return pd.DataFrame({
        "group": ["within", "between"],
        "mean_distance": [float(distances[same].mean()), float(distances[~same].mean())],
        "median_distance": [float(np.median(distances[same])), float(np.median(distances[~same]))],
        "pairs": [int(same.sum()), int((~same).sum())],
    })


def covariance_spectrum(features: np.ndarray, targets: np.ndarray, out_dir: str | Path, max_rank: int = 120) -> pd.DataFrame:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    means = np.zeros((int(targets.max()) + 1, features.shape[1]), dtype=np.float64)
    for cls in np.unique(targets):
        means[int(cls)] = features[targets == cls].mean(axis=0)
    centered = features - means[targets]
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = (singular_values ** 2) / max(len(centered) - len(np.unique(targets)), 1)
    ranks = np.arange(1, len(eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _apply_paper_style(ax, grid_axis="y")
    ax.plot(ranks[:max_rank], eigenvalues[:max_rank], color="#3B6FB6", linewidth=2.4)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Shared covariance eigenvalue")
    ax.set_title("Shared Covariance Spectrum", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    _finalize_diagnostic(fig, out_dir, "covariance_spectrum")
    return pd.DataFrame({"rank": ranks, "eigenvalue": eigenvalues})


def reliability_curve(predictions: pd.DataFrame, logits: np.ndarray, out_dir: str | Path, n_bins: int = 10) -> pd.DataFrame:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == predictions["target"].to_numpy(dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if mask.any():
            rows.append({
                "bin_left": left,
                "bin_right": right,
                "confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
            })
    df = pd.DataFrame(rows)
    ece = float(((df["count"] / len(confidence)) * (df["accuracy"] - df["confidence"]).abs()).sum()) if not df.empty else 0.0
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    _apply_paper_style(ax, grid_axis="both")
    ax.plot([0, 1], [0, 1], color="#111111", linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax.plot(df["confidence"], df["accuracy"], marker="o", color="#3B6FB6", linewidth=2.4, label="Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Curve", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False, loc="upper left")
    _finalize_diagnostic(fig, out_dir, "reliability_curve")
    df["ece"] = ece
    return df


def _add_cov_ellipse(ax, points: np.ndarray, color) -> None:
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if np.any(vals <= 0):
        return
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(vals)
    center = points.mean(axis=0)
    ellipse = Ellipse(center, width=width, height=height, angle=angle, facecolor="none", edgecolor=color, linewidth=1.0, alpha=0.65)
    ax.add_patch(ellipse)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def confusion_matrix_pair(zero_cm_path: str | Path, gda_cm_path: str | Path, out_dir: str | Path) -> None:
    zero_df = pd.read_csv(zero_cm_path, index_col=0)
    gda_df = pd.read_csv(gda_cm_path, index_col=0)
    labels = [str(label) for label in zero_df.index]
    matrices = []
    for cm_df in (zero_df, gda_df):
        cm = cm_df.values.astype(float)
        matrices.append(np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) > 0))
    vmax = max(float(m.max()) for m in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), constrained_layout=True)
    fig.set_facecolor(PAPER_BG)
    titles = ["Zero-shot CLIP", "CLIP + GDA"]
    im = None
    for ax, matrix, title in zip(axes, matrices, titles):
        ax.set_facecolor(PAPER_BG)
        im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=AXIS_TEXT, pad=12)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.4)
        ax.set_yticklabels(labels, fontsize=7.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
        cbar.set_label("Row-normalized frequency", color=AXIS_TEXT)
    fig.suptitle("Confusion Shift After Few-Shot GDA", fontsize=15, fontweight="bold", color=AXIS_TEXT, x=0.02, ha="left")
    _finalize_diagnostic(fig, out_dir, "confusion_zero_shot_vs_gda")


def per_class_accuracy_delta(zero_per_class: str | Path, gda_per_class: str | Path, out_dir: str | Path) -> pd.DataFrame:
    zero = pd.read_csv(zero_per_class)[["class_name", "accuracy"]].rename(columns={"accuracy": "zero_shot_accuracy"})
    gda = pd.read_csv(gda_per_class)[["class_name", "accuracy"]].rename(columns={"accuracy": "gda_accuracy"})
    df = zero.merge(gda, on="class_name")
    df["delta"] = df["gda_accuracy"] - df["zero_shot_accuracy"]
    df = df.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _apply_paper_style(ax, grid_axis="x")
    colors = np.where(df["delta"] >= 0, "#2F8F6B", "#C43B4D")
    bars = ax.barh(df["class_name"], df["delta"], color=colors, edgecolor="#222222", linewidth=0.35)
    ax.axvline(0, color="#111111", linewidth=0.9)
    ax.set_xlabel("Top-1 accuracy change")
    ax.set_title("Per-Class Accuracy Gain from GDA", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    for bar, value in zip(bars, df["delta"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.012 if value >= 0 else -0.012
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", ha=ha, va="center", fontsize=8.8, color=AXIS_TEXT)
    _finalize_diagnostic(fig, out_dir, "gda_delta_by_class")
    return df


def gda_validation_heatmap(gda_stats: dict, out_dir: str | Path) -> pd.DataFrame:
    scores = gda_stats.get("joint_validation_scores", {})
    rows = []
    for key, value in scores.items():
        parts = dict(part.split("=") for part in key.split("|") if "=" in part)
        rows.append({
            "covariance": parts.get("cov", ""),
            "ridge": float(parts.get("ridge", 0.0)),
            "alpha": float(parts.get("alpha", 0.0)),
            "validation_accuracy": float(value),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ridge_df = df[df["covariance"] == "ridge"].copy()
    if ridge_df.empty:
        ridge_df = df.copy()
    pivot = ridge_df.pivot_table(index="ridge", columns="alpha", values="validation_accuracy", aggfunc="max").sort_index()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.set_facecolor(PAPER_BG)
    fig.set_facecolor(PAPER_BG)
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{a:.1f}" for a in pivot.columns], fontsize=8.5)
    ax.set_yticklabels([f"{r:g}" for r in pivot.index], fontsize=8.5)
    ax.set_xlabel("GDA ensemble weight alpha")
    ax.set_ylabel("Ridge strength")
    ax.set_title("Validation Search for Regularized GDA", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    best = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
    ax.scatter(best[1], best[0], s=130, facecolors="none", edgecolors="#C43B4D", linewidths=2.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Validation accuracy", color=AXIS_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _finalize_diagnostic(fig, out_dir, "gda_validation_heatmap")
    return df


def _true_rank_from_top5(row: pd.Series) -> int:
    top5 = str(row.get("top5_predictions", "")).split("|")
    target = str(row.get("target_name", ""))
    for idx, pred in enumerate(top5, start=1):
        if pred == target:
            return idx
    return 6


def comparison_barplot(summary: pd.DataFrame, out_dir: str | Path) -> None:
    summary = _focus_dataset(summary)
    fewshot = summary[summary["Shots"] > 0]
    latest_shots = int(fewshot["Shots"].max()) if not fewshot.empty else int(summary["Shots"].max())
    df = summary[summary["Shots"] == latest_shots].copy()
    full_data = summary[(summary["Shots"] <= 0) & summary["Model"].str.contains("Full Fine-Tune", regex=False)].copy()
    df = pd.concat([df, full_data], ignore_index=True).drop_duplicates(["Model"], keep="last").sort_values("Top-1 Acc")
    df["Label"] = df["Model"].map(_short_model_name)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _apply_paper_style(ax, grid_axis="x")
    colors = [MODEL_COLORS.get(m, "#777777") for m in df["Model"]]
    bars = ax.barh(df["Label"], df["Top-1 Acc"], color=colors, height=0.58, edgecolor="#222222", linewidth=0.4)
    best_idx = int(np.argmax(df["Top-1 Acc"].to_numpy()))
    bars[best_idx].set_linewidth(1.8)
    bars[best_idx].set_edgecolor("#111111")
    xmax = min(1.0, max(0.25, float(df["Top-1 Acc"].max()) + 0.12))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Top-1 accuracy")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylabel("")
    title = f"{_dataset_title(summary)}: Model Comparison"
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    for bar, value in zip(bars, df["Top-1 Acc"]):
        ax.text(value + xmax * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="left", va="center", fontsize=10, color=AXIS_TEXT)
    ax.text(
        0.98,
        0.05,
        "Higher is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=MUTED_TEXT,
    )
    save_all_formats(fig, out_dir, "accuracy_comparison_barplot")


def training_free_clip_family_barplot(summary: pd.DataFrame, out_dir: str | Path) -> None:
    summary = _focus_dataset(summary)
    wanted = [
        ("Zero-shot CLIP ViT-B/16", "Zero-shot CLIP"),
        ("Zero-shot RemoteCLIP ViT-B/32", "Zero-shot RemoteCLIP"),
        ("CLIP ViT-B/16-QuickGELU + GDA", "CLIP + GDA"),
        ("RemoteCLIP ViT-B/32 + GDA", "RemoteCLIP + GDA"),
    ]
    rows = []
    for model, label in wanted:
        candidates = summary[summary["Model"] == model].sort_values(["Shots", "Top-1 Acc"], ascending=[False, False])
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        rows.append({"model": model, "label": label, "accuracy": float(row["Top-1 Acc"])})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("accuracy")

    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    _apply_paper_style(ax, grid_axis="x")
    colors = [MODEL_COLORS.get(model, "#777777") for model in df["model"]]
    bars = ax.barh(df["label"], df["accuracy"], color=colors, height=0.58, edgecolor="#222222", linewidth=0.4)
    xmax = min(1.0, max(0.45, float(df["accuracy"].max()) + 0.08))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Top-1 accuracy")
    ax.set_ylabel("")
    ax.set_title("Training-Free CLIP Backbone Comparison", loc="left", fontsize=14, fontweight="bold", color=AXIS_TEXT, pad=12)
    for bar, value in zip(bars, df["accuracy"]):
        ax.text(value + xmax * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="left", va="center", fontsize=9.5, color=AXIS_TEXT)
    save_all_formats(fig, out_dir, "summary_clip_family_barplot")


def accuracy_vs_shots(summary: pd.DataFrame, out_dir: str | Path) -> None:
    summary = _focus_dataset(summary)
    summary = summary[summary["Shots"] > 0].copy()
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _apply_paper_style(ax, grid_axis="y")
    label_rows = []
    for model, group in summary.groupby("Model"):
        group = group.sort_values("Shots")
        color = MODEL_COLORS.get(model, "#777777")
        if len(group) > 1:
            ax.plot(
                group["Shots"],
                group["Top-1 Acc"],
                marker="o",
                markersize=6,
                markeredgecolor=PAPER_BG,
                markeredgewidth=1.2,
                linewidth=2.5,
                label=model,
                color=color,
                path_effects=[pe.Stroke(linewidth=4.2, foreground=PAPER_BG), pe.Normal()],
            )
        else:
            ax.scatter(
                group["Shots"],
                group["Top-1 Acc"],
                marker="D",
                s=72,
                color=color,
                edgecolor=PAPER_BG,
                linewidth=1.2,
                zorder=4,
                label=model,
            )
        last = group.iloc[-1]
        label_rows.append({
            "x": float(last["Shots"]),
            "y": float(last["Top-1 Acc"]),
            "label": _short_model_name(model),
            "color": color,
        })
    ax.set_xscale("log", base=2)
    shots = sorted(summary["Shots"].unique())
    ax.set_xticks(shots)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(min(shots) * 0.82, max(shots) * 2.2)
    ax.set_xlabel("Shots per class")
    ax.set_ylabel("Top-1 accuracy")
    upper = min(1.02, max(0.25, float(summary["Top-1 Acc"].max()) + 0.12))
    ax.set_ylim(0, upper)
    ax.set_title(f"{_dataset_title(summary)}: Accuracy vs. Support Size", loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    label_y = _spread_positions([row["y"] for row in label_rows], lower=0.06, upper=upper - 0.04, min_gap=0.04)
    label_x = max(shots) * 1.12
    for row, y_adj in zip(label_rows, label_y):
        if abs(y_adj - row["y"]) > 1e-4:
            ax.plot([row["x"], label_x * 0.98], [row["y"], y_adj], color=row["color"], linewidth=0.8, alpha=0.45)
        ax.text(label_x, y_adj, row["label"], ha="left", va="center", fontsize=9.0, color=row["color"])
    save_all_formats(fig, out_dir, "accuracy_vs_shots_plot")


def confusion_matrix_plot(cm_path: str | Path, out_dir: str | Path, name: str = "confusion_matrix_gda", title: str = "Normalized Confusion Matrix: CLIP + GDA") -> None:
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm = cm_df.values.astype(float)
    norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.set_facecolor(PAPER_BG)
    fig.set_facecolor(PAPER_BG)
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=max(0.01, norm.max()), interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=16)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    labels = [str(label) for label in cm_df.index]
    if len(labels) <= 20:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized frequency", color=AXIS_TEXT)
    cbar.ax.tick_params(colors=AXIS_TEXT)
    save_all_formats(fig, out_dir, name)


def per_class_performance_plot(
    per_class_path: str | Path,
    out_dir: str | Path,
    title: str = "Per-Class Performance: CLIP + GDA",
    name: str = "per_class_performance_plot",
) -> None:
    df = pd.read_csv(per_class_path).sort_values("accuracy")
    if len(df) > 24:
        hardest = df.head(12).copy()
        easiest = df.tail(12).copy()
        hardest["group"] = "Hardest"
        easiest["group"] = "Easiest"
        plot_df = pd.concat([hardest, easiest], ignore_index=True)
    else:
        plot_df = df.copy()
        plot_df["group"] = ""
    plot_df = plot_df.sort_values(["group", "accuracy"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig_height = max(5.2, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    _apply_paper_style(ax, grid_axis="x")
    ax.hlines(y, plot_df["accuracy"], plot_df["f1"], color="#B8B1A7", linewidth=2.0, alpha=0.9)
    ax.scatter(plot_df["accuracy"], y, s=46, color="#C43B4D", label="Accuracy", zorder=3, edgecolor=PAPER_BG, linewidth=0.8)
    ax.scatter(plot_df["f1"], y, s=46, color="#3B6FB6", label="F1", zorder=3, edgecolor=PAPER_BG, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["class_name"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=AXIS_TEXT, pad=18)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), borderaxespad=0)
    save_all_formats(fig, out_dir, name)


def qualitative_grid(
    merged_predictions: pd.DataFrame,
    out_dir: str | Path,
    max_examples: int = 9,
) -> None:
    examples = _select_qualitative_examples(merged_predictions, max_examples)
    if examples.empty:
        return
    n = len(examples)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.35))
    fig.set_facecolor(PAPER_BG)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
        try:
            image = Image.open(row["path"]).convert("RGB")
            ax.imshow(image)
        except OSError:
            ax.set_facecolor("#EFEAE2")
            ax.text(0.5, 0.5, "Image unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=MUTED_TEXT)
        ax.axis("off")
        border_color = "#C43B4D"
        caption = "Confusion pair"
        if not row["gda_correct"]:
            caption = f"{row['target_name']} -> {row['gda_prediction_name']}"
        elif (not row["zero_correct"]) and row["gda_correct"]:
            border_color = "#2F8F6B"
            caption = "GDA improved"
        elif (not row["zero_correct"]) and row["convnext_correct"]:
            border_color = "#3B6FB6"
            caption = "ConvNeXt recovered"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(2.4)
        ax.text(0.02, 0.98, caption, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="white", bbox=dict(facecolor=border_color, edgecolor="none", pad=3.0, alpha=0.92))
        title = (
            f"GT: {row['target_name']}\n"
            f"Zero-shot: {row['zero_prediction_name']} ({row['zero_confidence']:.2f})\n"
            f"RemoteCLIP+GDA: {row['gda_prediction_name']} ({row['gda_confidence']:.2f})\n"
            f"ConvNeXt: {row['convnext_prediction_name']} ({row['convnext_confidence']:.2f})"
        )
        ax.set_title(title, fontsize=8.2, color=AXIS_TEXT, loc="left", pad=7)
    fig.suptitle("Qualitative Prediction Examples", fontsize=15, fontweight="bold", color=AXIS_TEXT, x=0.02, ha="left")
    save_all_formats(fig, out_dir, "qualitative_examples_grid")


def _select_qualitative_examples(df: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    wrong = df[~df["gda_correct"]].copy()
    if wrong.empty:
        return df.sort_values("gda_confidence", ascending=True).head(max_examples)

    pair_counts = (
        wrong.groupby(["target_name", "gda_prediction_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "target_name", "gda_prediction_name"], ascending=[False, True, True])
    )
    selected = []
    used_paths: set[str] = set()
    for _, pair in pair_counts.iterrows():
        candidates = wrong[
            (wrong["target_name"] == pair["target_name"])
            & (wrong["gda_prediction_name"] == pair["gda_prediction_name"])
        ].sort_values(["gda_confidence", "zero_confidence"], ascending=[False, False])
        for _, row in candidates.iterrows():
            if row["path"] not in used_paths:
                selected.append(row)
                used_paths.add(str(row["path"]))
                break
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        fallback = wrong.sort_values(["gda_confidence", "zero_confidence"], ascending=[False, False])
        for _, row in fallback.iterrows():
            if row["path"] not in used_paths:
                selected.append(row)
                used_paths.add(str(row["path"]))
            if len(selected) >= max_examples:
                break
    return pd.DataFrame(selected).head(max_examples)
