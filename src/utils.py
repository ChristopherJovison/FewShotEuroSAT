from __future__ import annotations

import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "dataset"


def get_dataset_name(config: dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {})
    return str(dataset_cfg.get("name", "eurosat"))


def make_run_dir(output_root: str | Path, method: str, shots: int, seed: int, dataset_name: str = "eurosat") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_slug = slugify(dataset_name)
    run_dir = Path(output_root) / dataset_slug / f"{method}_shot{shots}_seed{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "tables").mkdir()
    latest = Path(output_root) / dataset_slug / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)
    return run_dir


def refresh_latest(output_root: str | Path, run_dir: str | Path, dataset_name: str = "eurosat") -> None:
    latest = Path(output_root) / slugify(dataset_name) / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("eurosat_gda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run_log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def update_config_from_cli(config: dict[str, Any], shots: int | None, seed: int | None) -> dict[str, Any]:
    cfg = dict(config)
    cfg["experiment"] = dict(config["experiment"])
    if shots is not None:
        cfg["experiment"]["shots"] = shots
    if seed is not None:
        cfg["experiment"]["seed"] = seed
    return cfg


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    topk = np.argpartition(-logits, kth=min(k, logits.shape[1]) - 1, axis=1)[:, :k]
    return float(np.mean([target in row for target, row in zip(targets, topk)]))
