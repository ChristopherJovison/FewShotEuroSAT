from __future__ import annotations

import numpy as np


def one_hot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    values = np.zeros((len(targets), num_classes), dtype=np.float64)
    values[np.arange(len(targets)), targets] = 1.0
    return values


def tip_adapter_logits(
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_targets: np.ndarray,
    num_classes: int,
    beta: float,
) -> np.ndarray:
    query_features = np.asarray(query_features, dtype=np.float64)
    support_features = np.asarray(support_features, dtype=np.float64)
    affinity = query_features @ support_features.T
    cache_weights = np.exp(-float(beta) * (1.0 - affinity))
    cache_values = one_hot(support_targets, num_classes)
    return cache_weights @ cache_values


def combine_tip_logits(clip_logits: np.ndarray, cache_logits: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(clip_logits, dtype=np.float64) + float(alpha) * np.asarray(cache_logits, dtype=np.float64)


def prototype_logits(
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_targets: np.ndarray,
    num_classes: int,
    temperature: float = 1.0,
) -> np.ndarray:
    query_features = _l2_normalize(np.asarray(query_features, dtype=np.float64))
    support_features = _l2_normalize(np.asarray(support_features, dtype=np.float64))
    prototypes = np.zeros((num_classes, support_features.shape[1]), dtype=np.float64)
    for cls in range(num_classes):
        cls_features = support_features[np.asarray(support_targets) == cls]
        if len(cls_features) == 0:
            raise ValueError(f"Missing support examples for class {cls}.")
        prototypes[cls] = cls_features.mean(axis=0)
    prototypes = _l2_normalize(prototypes)
    return float(temperature) * (query_features @ prototypes.T)


def choose_three_way_fusion(
    clip_val_logits: np.ndarray,
    gda_val_logits: np.ndarray,
    proto_val_logits: np.ndarray,
    val_targets: np.ndarray,
    weight_grid: list[float],
) -> tuple[dict[str, float], dict[str, float]]:
    sources = {
        "clip": _standardize_logits(clip_val_logits),
        "gda": _standardize_logits(gda_val_logits),
        "prototype": _standardize_logits(proto_val_logits),
    }
    best_weights = {"clip": 0.0, "gda": 1.0, "prototype": 0.0}
    best_acc = -1.0
    scores: dict[str, float] = {}
    grid = [float(w) for w in weight_grid]
    for clip_w in grid:
        for gda_w in grid:
            for proto_w in grid:
                total = clip_w + gda_w + proto_w
                if total <= 0:
                    continue
                weights = {"clip": clip_w / total, "gda": gda_w / total, "prototype": proto_w / total}
                logits = fuse_three_way_logits(clip_val_logits, gda_val_logits, proto_val_logits, weights)
                acc = float((logits.argmax(axis=1) == val_targets).mean())
                key = f"clip={weights['clip']:.3f}|gda={weights['gda']:.3f}|prototype={weights['prototype']:.3f}"
                scores[key] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights
    return best_weights, scores


def fuse_three_way_logits(
    clip_logits: np.ndarray,
    gda_logits: np.ndarray,
    proto_logits: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    return (
        float(weights.get("clip", 0.0)) * _standardize_logits(clip_logits)
        + float(weights.get("gda", 0.0)) * _standardize_logits(gda_logits)
        + float(weights.get("prototype", 0.0)) * _standardize_logits(proto_logits)
    )


def choose_tip_hyperparams(
    clip_val_logits: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    support_features: np.ndarray,
    support_targets: np.ndarray,
    num_classes: int,
    alpha_grid: list[float],
    beta_grid: list[float],
) -> tuple[float, float, dict[str, float]]:
    best_alpha, best_beta, best_acc = float(alpha_grid[0]), float(beta_grid[0]), -1.0
    scores: dict[str, float] = {}
    for beta in beta_grid:
        cache_logits = tip_adapter_logits(val_features, support_features, support_targets, num_classes, float(beta))
        for alpha in alpha_grid:
            logits = combine_tip_logits(clip_val_logits, cache_logits, float(alpha))
            acc = float((logits.argmax(axis=1) == val_targets).mean())
            key = f"alpha={float(alpha):.3f},beta={float(beta):.3f}"
            scores[key] = acc
            if acc > best_acc:
                best_alpha, best_beta, best_acc = float(alpha), float(beta), acc
    return best_alpha, best_beta, scores


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def _standardize_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mean = logits.mean(axis=1, keepdims=True)
    std = logits.std(axis=1, keepdims=True)
    return (logits - mean) / np.maximum(std, 1e-8)
