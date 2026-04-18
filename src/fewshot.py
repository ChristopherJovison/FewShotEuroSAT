from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import json
import numpy as np


def sample_fewshot_indices(dataset: Any, shots: int, seed: int) -> dict[int, list[int]]:
    if shots <= 0:
        raise ValueError("shots must be positive.")
    by_class: dict[int, list[int]] = defaultdict(list)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [dataset[idx]["target"] for idx in range(len(dataset))]
    for idx, target in enumerate(targets):
        by_class[int(target)].append(idx)
    return sample_fewshot_indices_from_targets(targets, shots, seed)


def sample_fewshot_indices_from_targets(targets: list[int], shots: int, seed: int) -> dict[int, list[int]]:
    if shots <= 0:
        raise ValueError("shots must be positive.")
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        by_class[int(target)].append(idx)
    rng = np.random.default_rng(seed)
    selected: dict[int, list[int]] = {}
    for cls, indices in sorted(by_class.items()):
        if len(indices) < shots:
            raise ValueError(f"Class {cls} has only {len(indices)} examples, need {shots}.")
        selected[cls] = sorted(rng.choice(indices, size=shots, replace=False).astype(int).tolist())
    return selected


def flatten_support_indices(indices_by_class: dict[int, list[int]]) -> list[int]:
    return [idx for _, indices in sorted(indices_by_class.items()) for idx in indices]


def make_support_subset(dataset: Any, support_indices: dict[int, list[int]]):
    from torch.utils.data import Subset

    return Subset(dataset, flatten_support_indices(support_indices))


def support_metadata(
    support_indices: dict[int, list[int]],
    class_names: list[str],
    shots: int,
    seed: int,
    split: str = "train",
) -> dict[str, Any]:
    return {
        "dataset": "few-shot image classification",
        "split": split,
        "shots": shots,
        "seed": seed,
        "indices_by_class": {
            str(cls): {"class_name": class_names[cls], "indices": indices}
            for cls, indices in sorted(support_indices.items())
        },
    }


def save_support_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
