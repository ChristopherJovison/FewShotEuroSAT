from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import OAS


@dataclass
class GDAClassifier:
    means: np.ndarray
    precision: np.ndarray
    priors: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        linear = features @ self.precision @ self.means.T
        quadratic = -0.5 * np.sum((self.means @ self.precision) * self.means, axis=1)
        return linear + quadratic + np.log(self.priors + 1e-12)


def fit_gda(
    features: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ridge: float = 0.1,
    covariance: str = "ridge",
) -> GDAClassifier:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    dim = features.shape[1]
    means = np.zeros((num_classes, dim), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        cls_features = features[targets == cls]
        if len(cls_features) == 0:
            raise ValueError(f"Missing support examples for class {cls}.")
        means[cls] = cls_features.mean(axis=0)
        counts[cls] = len(cls_features)

    centered = features - means[targets]
    if covariance == "ridge":
        empirical_cov = centered.T @ centered / max(len(features) - num_classes, 1)
        trace = float(np.trace(empirical_cov) / dim)
        ridge_scale = ridge * trace if trace > 0 else ridge
        shrunk_cov = empirical_cov + ridge_scale * np.eye(dim)
        precision = np.linalg.pinv(shrunk_cov, hermitian=True)
    elif covariance == "oas":
        estimator = OAS(assume_centered=True, store_precision=True)
        estimator.fit(centered)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
    else:
        raise ValueError("covariance must be 'ridge' or 'oas'.")
    priors = counts / counts.sum()
    return GDAClassifier(means=means, precision=precision, priors=priors)


def choose_gda_hyperparams(
    support_features: np.ndarray,
    support_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    val_clip_logits: np.ndarray,
    num_classes: int,
    ridge_grid: list[float],
    alpha_grid: list[float],
    covariance_options: list[str],
) -> tuple[GDAClassifier, float, float, str, dict[str, float]]:
    best: tuple[float, GDAClassifier, float, float, str] | None = None
    scores: dict[str, float] = {}
    for covariance in covariance_options:
        ridges = ridge_grid if covariance == "ridge" else [0.0]
        for ridge in ridges:
            gda = fit_gda(support_features, support_targets, num_classes, ridge=float(ridge), covariance=covariance)
            val_gda_logits = gda.logits(val_features)
            for alpha in alpha_grid:
                logits = ensemble_logits(val_clip_logits, val_gda_logits, float(alpha))
                acc = float((logits.argmax(axis=1) == val_targets).mean())
                key = f"cov={covariance}|ridge={float(ridge):.6g}|alpha={float(alpha):.3f}"
                scores[key] = acc
                if best is None or acc > best[0]:
                    best = (acc, gda, float(ridge), float(alpha), covariance)
    if best is None:
        raise ValueError("No GDA hyperparameter candidates were evaluated.")
    _, best_gda, best_ridge, best_alpha, best_covariance = best
    return best_gda, best_ridge, best_alpha, best_covariance, scores


def ensemble_logits(clip_logits: np.ndarray, gda_logits: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    clip_scaled = _standardize_logits(clip_logits)
    gda_scaled = _standardize_logits(gda_logits)
    return (1.0 - alpha) * clip_scaled + alpha * gda_scaled


def choose_alpha(
    clip_val_logits: np.ndarray,
    gda_val_logits: np.ndarray,
    val_targets: np.ndarray,
    alpha_grid: list[float],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    best_alpha, best_acc = float(alpha_grid[0]), -1.0
    for alpha in alpha_grid:
        logits = ensemble_logits(clip_val_logits, gda_val_logits, float(alpha))
        acc = float((logits.argmax(axis=1) == val_targets).mean())
        scores[f"{float(alpha):.3f}"] = acc
        if acc > best_acc:
            best_alpha, best_acc = float(alpha), acc
    return best_alpha, scores


def _standardize_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mean = logits.mean(axis=1, keepdims=True)
    std = logits.std(axis=1, keepdims=True)
    return (logits - mean) / np.maximum(std, 1e-8)
