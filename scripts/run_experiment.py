from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_models import build_zeroshot_weights, clip_logits, extract_clip_features, load_clip, save_logits
from src.convnext_baseline import build_convnext_head, convnext_train_transforms, convnext_transforms, extract_convnext_features, fit_linear_head, predict_convnext_full, predict_with_head, train_convnext_full
from src.data import build_datasets
from src.fewshot import sample_fewshot_indices_from_targets, save_support_metadata, support_metadata
from src.gda import choose_alpha, choose_gda_hyperparams, ensemble_logits, fit_gda
from src.metrics import compute_metrics, predictions_frame, save_metric_outputs
from src.tip_adapter import choose_three_way_fusion, choose_tip_hyperparams, combine_tip_logits, fuse_three_way_logits, prototype_logits, tip_adapter_logits
from src.utils import get_dataset_name, make_run_dir, refresh_latest, resolve_device, save_json, save_yaml, set_seed, setup_logger, slugify, update_config_from_cli, load_yaml
from src.visuals import confusion_matrix_plot, per_class_performance_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EuroSAT few-shot experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--shots", type=int, default=None, help="Override shots per class.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    config = update_config_from_cli(load_yaml(args.config), args.shots, args.seed)
    exp = config["experiment"]
    dataset_name = get_dataset_name(config)
    set_seed(int(exp["seed"]))
    device = resolve_device(str(exp.get("device", "auto")))
    run_dir = make_run_dir(exp["output_root"], exp["method"], int(exp["shots"]), int(exp["seed"]), dataset_name)
    logger = setup_logger(run_dir)
    save_yaml(config, run_dir / "config_used.yaml")
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    start = time.perf_counter()

    method = exp["method"]
    if method in {"zeroshot_clip", "clip_gda", "clip_tip_adapter", "clip_gda_proto"}:
        logits, targets, indices, paths, extra_metrics, class_names = run_clip_method(config, device, run_dir, logger)
    elif method == "convnext_tiny":
        logits, targets, indices, paths, extra_metrics, class_names = run_convnext_method(config, device, run_dir, logger)
    elif method == "convnext_tiny_full_finetune":
        logits, targets, indices, paths, extra_metrics, class_names = run_convnext_full_finetune(config, device, run_dir, logger)
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics, per_class, confusion = compute_metrics(logits, targets, class_names)
    metrics.update(extra_metrics)
    metrics["dataset"] = dataset_name
    metrics["runtime_seconds"] = round(time.perf_counter() - start, 2)
    preds = predictions_frame(logits, targets, indices, paths, class_names)
    save_metric_outputs(metrics, preds, per_class, confusion, run_dir)
    per_class.to_csv(run_dir / "tables" / "per_class_metrics.csv", index=False)
    confusion.to_csv(run_dir / "tables" / "confusion_matrix.csv")
    if method == "clip_gda":
        confusion_matrix_plot(run_dir / "confusion_matrix.csv", run_dir / "figures")
        per_class_performance_plot(run_dir / "per_class_metrics.csv", run_dir / "figures")
    refresh_latest(exp["output_root"], run_dir, dataset_name)
    logger.info("Top-1 accuracy: %.4f", metrics["top1_accuracy"])
    logger.info("Finished in %.2fs", metrics["runtime_seconds"])


def run_clip_method(config, device, run_dir: Path, logger):
    exp = config["experiment"]
    clip_cfg = config["clip"]
    gda_cfg = config.get("gda", {})
    dataset_name = get_dataset_name(config)
    cache_ready = clip_cache_ready(exp["data_root"], dataset_name, clip_cfg)
    needs_support_view_model = exp["method"] in {"clip_gda", "clip_gda_proto"} and bool(gda_cfg.get("support_feature_augmentations"))
    if cache_ready and not needs_support_view_model:
        logger.info("Using cached CLIP features and text weights; skipping CLIP model load.")
        model, tokenizer = None, None
        preprocess = None
    else:
        logger.info("Loading frozen CLIP %s (%s)", clip_cfg["model_name"], clip_cfg["pretrained"])
        model, preprocess, tokenizer = load_clip(
            clip_cfg["model_name"],
            clip_cfg.get("pretrained"),
            device,
            checkpoint_path=clip_cfg.get("checkpoint_path"),
            hf_repo_id=clip_cfg.get("hf_repo_id"),
            hf_filename=clip_cfg.get("hf_filename"),
            checkpoint_cache_dir=clip_cfg.get("checkpoint_cache_dir"),
        )
    train_base, val_data, test_data, class_names = build_datasets(config, transform=preprocess)
    pool_targets = train_base.targets + val_data.targets
    support_by_class = sample_fewshot_indices_from_targets(pool_targets, int(exp["shots"]), int(exp["seed"]))
    support_meta = support_metadata(support_by_class, class_names, int(exp["shots"]), int(exp["seed"]), split="train+val")
    support_meta["dataset"] = dataset_name
    save_support_metadata(run_dir / "support_indices.json", support_meta)
    zeroshot_weights = cached_zeroshot_weights(model, tokenizer, class_names, device, exp["data_root"], dataset_name, clip_cfg, logger)
    batch_size = int(clip_cfg.get("batch_size", 64))
    num_workers = int(clip_cfg.get("num_workers", 0))
    feature_augmentations = list(clip_cfg.get("feature_augmentations", ["identity"]))
    logger.info("CLIP feature augmentations: %s", ",".join(feature_augmentations))
    test_pack = cached_clip_features(model, test_data, device, batch_size, num_workers, feature_augmentations, exp["data_root"], dataset_name, clip_cfg, "test", logger)
    test_clip_logits = clip_logits(test_pack["features"], zeroshot_weights)
    save_logits(test_clip_logits, run_dir / "clip_zeroshot_logits.npy")
    if exp["method"] == "zeroshot_clip":
        return test_clip_logits, test_pack["targets"], test_pack["indices"], test_pack["paths"], {"adaptation": "none"}, class_names

    train_pack = cached_clip_features(model, train_base, device, batch_size, num_workers, feature_augmentations, exp["data_root"], dataset_name, clip_cfg, "train", logger)
    val_pack = cached_clip_features(model, val_data, device, batch_size, num_workers, feature_augmentations, exp["data_root"], dataset_name, clip_cfg, "val", logger)
    pool_features = np.concatenate([train_pack["features"], val_pack["features"]], axis=0)
    pool_targets_np = np.concatenate([train_pack["targets"], val_pack["targets"]], axis=0)
    support_rows = np.array([idx for indices in support_by_class.values() for idx in indices], dtype=int)
    support_features = pool_features[support_rows]
    support_targets = pool_targets_np[support_rows]
    val_keep = validation_keep_mask(len(train_pack["targets"]), len(val_pack["targets"]), support_rows)
    val_features = val_pack["features"][val_keep]
    val_targets = val_pack["targets"][val_keep]
    if exp["method"] == "clip_tip_adapter":
        logits, targets, indices, paths, extra_metrics = run_tip_adapter(
            config,
            run_dir,
            logger,
            zeroshot_weights,
            test_pack,
            test_clip_logits,
            support_features,
            support_targets,
            val_features,
            val_targets,
            class_names,
        )
        return logits, targets, indices, paths, extra_metrics, class_names

    gda_cfg = config["gda"]
    support_feature_augmentations = gda_cfg.get("support_feature_augmentations")
    if support_feature_augmentations:
        if model is None:
            raise RuntimeError("Support-view expansion requires a loaded CLIP model.")
        support_features, support_targets = extract_support_view_features(
            model,
            train_base,
            val_data,
            support_rows,
            len(train_pack["targets"]),
            device,
            batch_size,
            num_workers,
            [str(aug) for aug in support_feature_augmentations],
            logger,
        )
        logger.info("Expanded GDA support features to %d rows using %d views.", len(support_targets), len(support_feature_augmentations))
    val_clip_logits = clip_logits(val_features, zeroshot_weights)
    alpha_grid = [float(a) for a in gda_cfg["alpha_grid"]]
    if gda_cfg.get("search_ridge", False) or gda_cfg.get("covariance_options"):
        gda, ridge, alpha, covariance, gda_scores = choose_gda_hyperparams(
            support_features,
            support_targets,
            val_features,
            val_targets,
            val_clip_logits,
            len(class_names),
            [float(r) for r in gda_cfg.get("ridge_grid", [gda_cfg.get("ridge", 0.1)])],
            alpha_grid,
            [str(c) for c in gda_cfg.get("covariance_options", [gda_cfg.get("covariance", "ridge")])],
        )
        alpha_scores = {}
    else:
        ridge = float(gda_cfg.get("ridge", 0.1))
        covariance = str(gda_cfg.get("covariance", "ridge"))
        gda = fit_gda(support_features, support_targets, len(class_names), ridge=ridge, covariance=covariance)
        val_gda_logits = gda.logits(val_features)
        if gda_cfg.get("alpha") is None:
            alpha, alpha_scores = choose_alpha(val_clip_logits, val_gda_logits, val_targets, alpha_grid)
        else:
            alpha, alpha_scores = float(gda_cfg["alpha"]), {}
        gda_scores = {}
    logger.info("Chosen GDA covariance=%s ridge=%.6g alpha=%.3f", covariance, ridge, alpha)
    save_json({"alpha": alpha, "alpha_validation_scores": alpha_scores, "ridge": ridge, "covariance": covariance, "joint_validation_scores": gda_scores}, run_dir / "gda_stats.json")
    val_gda_logits = gda.logits(val_features)
    test_gda_logits = gda.logits(test_pack["features"])
    if exp["method"] == "clip_gda_proto":
        proto_cfg = config.get("prototype_fusion", {})
        proto_temperature = float(proto_cfg.get("temperature", 1.0))
        val_proto_logits = prototype_logits(val_features, support_features, support_targets, len(class_names), temperature=proto_temperature)
        test_proto_logits = prototype_logits(test_pack["features"], support_features, support_targets, len(class_names), temperature=proto_temperature)
        fusion_weights, fusion_scores = choose_three_way_fusion(
            val_clip_logits,
            val_gda_logits,
            val_proto_logits,
            val_targets,
            [float(w) for w in proto_cfg.get("weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0])],
        )
        logits = fuse_three_way_logits(test_clip_logits, test_gda_logits, test_proto_logits, fusion_weights)
        np.save(run_dir / "prototype_logits.npy", test_proto_logits)
        save_json(
            {
                "weights": fusion_weights,
                "validation_scores": fusion_scores,
                "prototype_temperature": proto_temperature,
            },
            run_dir / "prototype_fusion_stats.json",
        )
        adaptation = "training-free_gda_prototype_fusion"
    else:
        logits = ensemble_logits(test_clip_logits, test_gda_logits, alpha)
        fusion_weights = None
        adaptation = "training-free_gda"
    np.save(run_dir / "gda_logits.npy", test_gda_logits)
    extra = {
        "adaptation": adaptation,
        "alpha": alpha,
        "ridge": ridge,
        "covariance": covariance,
    }
    if fusion_weights is not None:
        extra["fusion_weights"] = fusion_weights
    return logits, test_pack["targets"], test_pack["indices"], test_pack["paths"], extra, class_names


def run_tip_adapter(
    config,
    run_dir: Path,
    logger,
    zeroshot_weights,
    test_pack: dict,
    test_clip_logits: np.ndarray,
    support_features: np.ndarray,
    support_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    class_names: list[str],
):
    tip_cfg = config["tip_adapter"]
    val_clip_logits = clip_logits(val_features, zeroshot_weights)
    if tip_cfg.get("alpha") is None or tip_cfg.get("beta") is None:
        alpha, beta, search_scores = choose_tip_hyperparams(
            val_clip_logits,
            val_features,
            val_targets,
            support_features,
            support_targets,
            len(class_names),
            [float(a) for a in tip_cfg["alpha_grid"]],
            [float(b) for b in tip_cfg["beta_grid"]],
        )
    else:
        alpha, beta, search_scores = float(tip_cfg["alpha"]), float(tip_cfg["beta"]), {}
    logger.info("Chosen Tip-Adapter alpha: %.3f, beta: %.3f", alpha, beta)
    test_cache_logits = tip_adapter_logits(
        test_pack["features"],
        support_features,
        support_targets,
        len(class_names),
        beta,
    )
    logits = combine_tip_logits(test_clip_logits, test_cache_logits, alpha)
    np.save(run_dir / "tip_adapter_logits.npy", test_cache_logits)
    save_json({"alpha": alpha, "beta": beta, "validation_scores": search_scores}, run_dir / "tip_adapter_stats.json")
    return logits, test_pack["targets"], test_pack["indices"], test_pack["paths"], {
        "adaptation": "training-free_tip_adapter",
        "alpha": alpha,
        "beta": beta,
    }


def extract_support_view_features(
    model,
    train_data,
    val_data,
    support_rows: np.ndarray,
    train_len: int,
    device,
    batch_size: int,
    num_workers: int,
    augmentations: list[str],
    logger,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = [int(idx) for idx in support_rows if int(idx) < train_len]
    val_indices = [int(idx - train_len) for idx in support_rows if int(idx) >= train_len]
    features, targets = [], []
    for aug in augmentations:
        logger.info("Extracting support-view features for augmentation: %s", aug)
        if train_indices:
            train_pack = extract_clip_features(
                model,
                Subset(train_data, train_indices),
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                desc=f"CLIP support train features ({aug})",
                feature_augmentations=[aug],
            )
            features.append(train_pack["features"])
            targets.append(train_pack["targets"])
        if val_indices:
            val_pack = extract_clip_features(
                model,
                Subset(val_data, val_indices),
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                desc=f"CLIP support val features ({aug})",
                feature_augmentations=[aug],
            )
            features.append(val_pack["features"])
            targets.append(val_pack["targets"])
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def run_convnext_method(config, device, run_dir: Path, logger):
    exp = config["experiment"]
    conv_cfg = config["convnext"]
    dataset_name = get_dataset_name(config)
    transform = convnext_transforms(conv_cfg.get("weights", "IMAGENET1K_V1"))
    train_base, val_data, test_data, class_names = build_datasets(config, transform=transform)
    pool_targets = train_base.targets + val_data.targets
    support_by_class = sample_fewshot_indices_from_targets(pool_targets, int(exp["shots"]), int(exp["seed"]))
    support_meta = support_metadata(support_by_class, class_names, int(exp["shots"]), int(exp["seed"]), split="train+val")
    support_meta["dataset"] = dataset_name
    save_support_metadata(run_dir / "support_indices.json", support_meta)
    if convnext_cache_ready(exp["data_root"], dataset_name, conv_cfg):
        logger.info("Using cached ConvNeXt features; skipping ConvNeXt backbone load.")
        model = None
    else:
        model = build_convnext_head(len(class_names), conv_cfg.get("weights", "IMAGENET1K_V1")).to(device)
    batch_size = int(conv_cfg.get("batch_size", 64))
    num_workers = int(conv_cfg.get("num_workers", 0))
    train_features, train_targets, _, _, _ = cached_convnext_features(model, train_base, device, batch_size, num_workers, exp["data_root"], dataset_name, conv_cfg, "train", logger)
    val_features, val_targets, _, _, _ = cached_convnext_features(model, val_data, device, batch_size, num_workers, exp["data_root"], dataset_name, conv_cfg, "val", logger)
    test_features, test_targets, test_indices, test_paths, test_class_names = cached_convnext_features(model, test_data, device, batch_size, num_workers, exp["data_root"], dataset_name, conv_cfg, "test", logger)
    pool_features = torch.cat([train_features, val_features], dim=0)
    pool_targets_tensor = torch.cat([train_targets, val_targets], dim=0)
    support_rows = torch.tensor([idx for indices in support_by_class.values() for idx in indices], dtype=torch.long, device=device)
    val_keep_np = validation_keep_mask(train_features.shape[0], val_features.shape[0], support_rows.detach().cpu().numpy())
    val_keep = torch.tensor(val_keep_np, dtype=torch.bool, device=device)
    logger.info("Training ConvNeXt-Tiny %s head for %d epochs", conv_cfg.get("mode", "linear_probe"), int(conv_cfg.get("epochs", 50)))
    head = fit_linear_head(
        pool_features[support_rows],
        pool_targets_tensor[support_rows],
        val_features[val_keep],
        val_targets[val_keep],
        device,
        batch_size=batch_size,
        epochs=int(conv_cfg.get("epochs", 50)),
        lr=float(conv_cfg.get("lr", 1e-3)),
        weight_decay=float(conv_cfg.get("weight_decay", 1e-4)),
    )
    pack = predict_with_head(head, test_features, test_targets, test_indices, test_paths, test_class_names, batch_size=batch_size)
    return pack["logits"], pack["targets"], pack["indices"], pack["paths"], {
        "adaptation": "supervised_fewshot_head",
        "mode": conv_cfg.get("mode", "linear_probe"),
        "epochs": int(conv_cfg.get("epochs", 50)),
    }, class_names


def run_convnext_full_finetune(config, device, run_dir: Path, logger):
    exp = config["experiment"]
    conv_cfg = config["convnext"]
    dataset_name = get_dataset_name(config)
    weights_name = conv_cfg.get("weights", "IMAGENET1K_V1")
    train_transform = convnext_train_transforms(weights_name)
    eval_transform = convnext_transforms(weights_name)
    train_data, _, _, class_names = build_datasets(config, transform=train_transform)
    _, val_data, test_data, _ = build_datasets(config, transform=eval_transform)
    support_meta = {
        "dataset": dataset_name,
        "split": "full_train",
        "shots": "all",
        "seed": int(exp["seed"]),
        "num_train_examples": len(train_data),
        "num_val_examples": len(val_data),
        "note": "Full supervised fine-tuning uses the complete training split, not a few-shot support subset.",
    }
    save_json(support_meta, run_dir / "support_indices.json")
    logger.info("Loading ImageNet-pretrained ConvNeXt-Tiny for full supervised fine-tuning.")
    model = build_convnext_head(len(class_names), weights_name)
    model, history = train_convnext_full(
        model,
        train_data,
        val_data,
        device,
        batch_size=int(conv_cfg.get("batch_size", 32)),
        num_workers=int(conv_cfg.get("num_workers", 0)),
        epochs=int(conv_cfg.get("epochs", 5)),
        lr=float(conv_cfg.get("lr", 5e-5)),
        weight_decay=float(conv_cfg.get("weight_decay", 0.05)),
        label_smoothing=float(conv_cfg.get("label_smoothing", 0.05)),
    )
    save_json(history, run_dir / "training_history.json")
    torch.save(model.state_dict(), run_dir / "convnext_tiny_full_finetune.pt")
    pack = predict_convnext_full(
        model,
        test_data,
        device,
        batch_size=int(conv_cfg.get("batch_size", 32)),
        num_workers=int(conv_cfg.get("num_workers", 0)),
    )
    return pack["logits"], pack["targets"], pack["indices"], pack["paths"], {
        "adaptation": "supervised_full_finetune",
        "mode": "full_finetune",
        "epochs": int(conv_cfg.get("epochs", 5)),
        "best_val_accuracy": history["best_val_accuracy"],
    }, class_names


def cached_clip_features(model, dataset, device, batch_size: int, num_workers: int, feature_augmentations: list[str], data_root: str, dataset_name: str, clip_cfg: dict, split: str, logger):
    cache_path = clip_feature_cache_path(data_root, dataset_name, clip_cfg, split, feature_augmentations)
    if cache_path.exists():
        logger.info("Loading cached CLIP %s features: %s", split, cache_path)
        try:
            return load_npz_pack(cache_path)
        except Exception as exc:
            if model is None:
                raise
            logger.warning("Ignoring unreadable CLIP %s cache at %s (%s); recomputing.", split, cache_path, exc)
    if model is None:
        raise FileNotFoundError(f"Missing CLIP {split} feature cache: {cache_path}")
    pack = extract_clip_features(model, dataset, device, batch_size=batch_size, num_workers=num_workers, desc=f"CLIP {split} features", feature_augmentations=feature_augmentations)
    save_npz_pack(pack, cache_path)
    return pack


def validation_keep_mask(train_len: int, val_len: int, support_rows: np.ndarray) -> np.ndarray:
    val_support = support_rows[support_rows >= train_len] - train_len
    keep = np.ones(val_len, dtype=bool)
    keep[val_support.astype(int)] = False
    if not keep.any():
        raise ValueError("No validation examples remain after support sampling.")
    return keep


def cached_zeroshot_weights(model, tokenizer, class_names: list[str], device, data_root: str, dataset_name: str, clip_cfg: dict, logger):
    ensemble = bool(clip_cfg.get("prompt_ensemble", True))
    cache_path = clip_zeroshot_cache_path(data_root, dataset_name, clip_cfg, ensemble)
    if cache_path.exists():
        logger.info("Loading cached CLIP zero-shot text weights: %s", cache_path)
        return torch.tensor(np.load(cache_path), dtype=torch.float32, device=device)
    if model is None or tokenizer is None:
        raise FileNotFoundError(f"Missing CLIP zero-shot text cache: {cache_path}")
    weights = build_zeroshot_weights(
        model,
        tokenizer,
        class_names,
        device,
        ensemble,
        clip_cfg.get("prompt_templates"),
    )
    np.save(cache_path, weights.detach().cpu().numpy())
    return weights


def clip_cache_ready(data_root: str, dataset_name: str, clip_cfg: dict) -> bool:
    ensemble = bool(clip_cfg.get("prompt_ensemble", True))
    feature_augmentations = list(clip_cfg.get("feature_augmentations", ["identity"]))
    paths = [clip_feature_cache_path(data_root, dataset_name, clip_cfg, split, feature_augmentations) for split in ("train", "val", "test")]
    paths.append(clip_zeroshot_cache_path(data_root, dataset_name, clip_cfg, ensemble))
    return all(path.exists() for path in paths)


def clip_feature_cache_path(data_root: str, dataset_name: str, clip_cfg: dict, split: str, feature_augmentations: list[str] | None = None) -> Path:
    cache_dir = Path(data_root) / "feature_cache" / slugify(dataset_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    aug_suffix = augmentations_cache_suffix(feature_augmentations or ["identity"])
    suffix = f"_{aug_suffix}" if aug_suffix else ""
    return cache_dir / f"clip_{clip_cfg['model_name']}_{clip_cfg['pretrained']}_{split}{suffix}.npz".replace("/", "-")


def clip_zeroshot_cache_path(data_root: str, dataset_name: str, clip_cfg: dict, ensemble: bool) -> Path:
    cache_dir = Path(data_root) / "feature_cache" / slugify(dataset_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_suffix = prompt_cache_suffix(clip_cfg)
    return cache_dir / f"clip_{clip_cfg['model_name']}_{clip_cfg['pretrained']}_zeroshot_promptensemble{int(ensemble)}_{prompt_suffix}.npy".replace("/", "-")


def prompt_cache_suffix(clip_cfg: dict) -> str:
    templates = clip_cfg.get("prompt_templates")
    if not templates:
        return "default"
    encoded = json.dumps(templates, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def augmentations_cache_suffix(feature_augmentations: list[str]) -> str:
    if feature_augmentations == ["identity"]:
        return ""
    encoded = json.dumps(feature_augmentations).encode("utf-8")
    return "aug_" + hashlib.sha1(encoded).hexdigest()[:8]


def cached_convnext_features(model, dataset, device, batch_size: int, num_workers: int, data_root: str, dataset_name: str, conv_cfg: dict, split: str, logger):
    cache_path = convnext_feature_cache_path(data_root, dataset_name, conv_cfg, split)
    if cache_path.exists():
        logger.info("Loading cached ConvNeXt %s features: %s", split, cache_path)
        pack = load_npz_pack(cache_path)
        return (
            torch.tensor(pack["features"], dtype=torch.float32, device=device),
            torch.tensor(pack["targets"], dtype=torch.long, device=device),
            pack["indices"],
            pack["paths"],
            pack["class_names"],
        )
    if model is None:
        raise FileNotFoundError(f"Missing ConvNeXt {split} feature cache: {cache_path}")
    features, targets, indices, paths, class_names = extract_convnext_features(model, dataset, device, batch_size, f"ConvNeXt {split} features", num_workers=num_workers)
    pack = {
        "features": features.detach().cpu().numpy(),
        "targets": targets.detach().cpu().numpy(),
        "indices": indices,
        "paths": paths,
        "class_names": class_names,
    }
    save_npz_pack(pack, cache_path)
    return features, targets, indices, paths, class_names


def convnext_cache_ready(data_root: str, dataset_name: str, conv_cfg: dict) -> bool:
    paths = [convnext_feature_cache_path(data_root, dataset_name, conv_cfg, split) for split in ("train", "val", "test")]
    return all(path.exists() for path in paths)


def convnext_feature_cache_path(data_root: str, dataset_name: str, conv_cfg: dict, split: str) -> Path:
    cache_dir = Path(data_root) / "feature_cache" / slugify(dataset_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights = conv_cfg.get("weights", "IMAGENET1K_V1")
    return cache_dir / f"convnext_tiny_{weights}_{split}.npz"


def save_npz_pack(pack: dict, cache_path: Path) -> None:
    np.savez_compressed(
        cache_path,
        features=np.asarray(pack["features"]),
        targets=np.asarray(pack["targets"]),
        indices=np.asarray(pack["indices"], dtype=np.int64),
        paths=np.asarray(pack["paths"], dtype=object),
        class_names=np.asarray(pack["class_names"], dtype=object),
    )


def load_npz_pack(cache_path: Path) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    return {
        "features": data["features"],
        "targets": data["targets"],
        "indices": data["indices"].astype(int).tolist(),
        "paths": data["paths"].astype(str).tolist(),
        "class_names": data["class_names"].astype(str).tolist(),
    }


if __name__ == "__main__":
    main()
