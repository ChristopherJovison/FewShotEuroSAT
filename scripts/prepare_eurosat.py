from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


DATASET_SLUG = "waseemalastal/eurosat-rgb-dataset"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EuroSAT RGB from KaggleHub and prepare a deterministic ImageFolder split.")
    parser.add_argument("--output_dir", default="data/eurosat/imagefolder", help="Prepared ImageFolder output directory.")
    parser.add_argument("--metadata_path", default="data/eurosat/split_metadata.json", help="Path for split metadata JSON.")
    parser.add_argument("--raw_dir", default=None, help="Existing raw EuroSAT directory. If omitted, KaggleHub downloads the dataset.")
    parser.add_argument("--cache_dir", default="data/kagglehub", help="KaggleHub cache directory used when raw_dir is omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--force", action="store_true", help="Recreate output_dir even if it already exists.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else download_eurosat_with_kagglehub(Path(args.cache_dir))
    metadata = prepare_eurosat_imagefolder(
        raw_dir=raw_dir,
        output_dir=Path(args.output_dir),
        metadata_path=Path(args.metadata_path),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        force=args.force,
    )
    print(f"Prepared EuroSAT ImageFolder at {Path(args.output_dir).resolve()}")
    print(f"Images: {sum(row['total'] for row in metadata['classes'].values())}")


def download_eurosat_with_kagglehub(cache_dir: str | Path = "data/kagglehub", dataset_slug: str = DATASET_SLUG) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Install kagglehub to download EuroSAT: pip install kagglehub") from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir.resolve())
    return Path(kagglehub.dataset_download(dataset_slug))


def prepare_eurosat_imagefolder(
    raw_dir: str | Path,
    output_dir: str | Path = "data/eurosat/imagefolder",
    metadata_path: str | Path = "data/eurosat/split_metadata.json",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    force: bool = False,
) -> dict:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw EuroSAT directory not found: {raw_dir}")
    if output_dir.exists():
        if not force:
            return load_or_describe_existing_split(output_dir, metadata_path)
        shutil.rmtree(output_dir)

    class_dirs = find_class_dirs(raw_dir)
    rng = random.Random(seed)
    metadata = {
        "dataset": "eurosat",
        "source": DATASET_SLUG,
        "split": f"deterministic {int(train_ratio * 100)}/{int(val_ratio * 100)}/{int((1 - train_ratio - val_ratio) * 100)} per class",
        "seed": seed,
        "classes": {},
    }

    for class_name in EUROSAT_CLASSES:
        images = sorted(p for p in class_dirs[class_name].iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"No images found for class {class_name} in {class_dirs[class_name]}")
        rng.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split, paths in splits.items():
            split_class_dir = output_dir / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, split_class_dir / src.name)
        metadata["classes"][class_name] = {
            "total": n_total,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def find_class_dirs(raw_dir: Path) -> dict[str, Path]:
    class_dirs: dict[str, Path] = {}
    for class_name in EUROSAT_CLASSES:
        matches = [p for p in raw_dir.rglob(class_name) if p.is_dir()]
        matches = [p for p in matches if any(child.suffix.lower() in IMAGE_EXTENSIONS for child in p.iterdir() if child.is_file())]
        if not matches:
            raise FileNotFoundError(f"Could not find class directory {class_name!r} under {raw_dir}")
        class_dirs[class_name] = sorted(matches, key=lambda p: len(p.parts))[0]
    return class_dirs


def load_or_describe_existing_split(output_dir: Path, metadata_path: Path) -> dict:
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    classes = {}
    for class_name in EUROSAT_CLASSES:
        counts = {}
        for split in ("train", "val", "test"):
            class_dir = output_dir / split / class_name
            counts[split] = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if class_dir.exists() else 0
        classes[class_name] = {"total": sum(counts.values()), **counts}
    return {
        "dataset": "eurosat",
        "source": DATASET_SLUG,
        "split": "existing ImageFolder split",
        "seed": None,
        "classes": classes,
    }


if __name__ == "__main__":
    main()
