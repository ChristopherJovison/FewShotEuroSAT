from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def eurosat_sample_grid(
    dataset_dir: str | Path = "data/eurosat/imagefolder",
    output: str | Path = "paper_assets/eurosat_samples.png",
    split: str = "train",
    seed: int = 42,
    class_order: list[str] | None = None,
) -> Path:
    dataset_dir = Path(dataset_dir)
    output = Path(output)
    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    rng = random.Random(seed)
    samples = []
    for class_name in class_order or EUROSAT_CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"No images found in {class_dir}")
        samples.append((class_name, rng.choice(images)))

    output.parent.mkdir(parents=True, exist_ok=True)
    plot_image_grid(samples, output)
    return output


def plot_image_grid(samples: list[tuple[str, Path]], output: str | Path, rows: int = 2, cols: int = 5, figsize: tuple[int, int] = (12, 5)) -> None:
    output = Path(output)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for ax, (class_name, path) in zip(axes.ravel(), samples):
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.set_title(class_name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
