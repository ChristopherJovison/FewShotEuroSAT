from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


SPLITS = ("train", "val", "test")


class ImageFolderWithPaths(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        class_names: list[str] | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {SPLITS}.")
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing ImageFolder split directory: {split_root}")
        self.dataset = ImageFolder(str(split_root), transform=None)
        self.split = split
        self.transform = transform
        self.folder_classes = list(self.dataset.classes)
        if class_names is not None and len(class_names) != len(self.folder_classes):
            raise ValueError(
                f"Config class_names has {len(class_names)} entries, but ImageFolder found "
                f"{len(self.folder_classes)} classes in {split_root}."
            )
        self.class_names = class_names or [humanize_folder_name(name) for name in self.folder_classes]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def targets(self) -> list[int]:
        return [int(target) for target in self.dataset.targets]

    def __getitem__(self, index: int) -> dict[str, object]:
        path, target = self.dataset.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": int(target),
            "index": int(index),
            "path": str(path),
            "class_name": self.class_names[int(target)],
        }


def humanize_folder_name(name: str) -> str:
    text = name.replace("_", " ").replace("-", " ")
    chars: list[str] = []
    for idx, char in enumerate(text):
        prev = text[idx - 1] if idx else ""
        if idx and char.isupper() and prev.islower():
            chars.append(" ")
        chars.append(char)
    return " ".join("".join(chars).split()).lower()


def build_datasets(config: dict, transform: Callable | None = None) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    dataset_cfg = config.get("dataset", {"name": "eurosat", "type": "imagefolder"})
    dataset_type = dataset_cfg.get("type", "imagefolder")
    if dataset_type == "imagefolder":
        dataset_root = dataset_cfg.get("root")
        if dataset_root is None:
            raise ValueError("ImageFolder datasets require dataset.root in the config.")
        class_names = dataset_cfg.get("class_names")
        train_data = ImageFolderWithPaths(dataset_root, "train", transform=transform, class_names=class_names)
        val_data = ImageFolderWithPaths(dataset_root, "val", transform=transform, class_names=class_names)
        test_data = ImageFolderWithPaths(dataset_root, "test", transform=transform, class_names=class_names)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    class_names = list(getattr(train_data, "class_names"))
    return train_data, val_data, test_data, class_names


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[arg-type]
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "index": [int(item["index"]) for item in batch],
        "path": [str(item["path"]) for item in batch],
        "class_name": [str(item["class_name"]) for item in batch],
    }


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
