from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import collate_batch
from .prompts import build_prompts


@torch.no_grad()
def load_clip(
    model_name: str,
    pretrained: str | None,
    device: torch.device,
    checkpoint_path: str | None = None,
    hf_repo_id: str | None = None,
    hf_filename: str | None = None,
    checkpoint_cache_dir: str | None = None,
):
    if hf_repo_id and hf_filename:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            cache_dir=checkpoint_cache_dir,
        )
    if checkpoint_path:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess, tokenizer


@torch.no_grad()
def build_zeroshot_weights(
    model,
    tokenizer,
    class_names: list[str],
    device: torch.device,
    prompt_ensemble: bool = True,
    prompt_templates: list[str] | None = None,
) -> torch.Tensor:
    prompts_by_class = build_prompts(
        class_names,
        ensemble=prompt_ensemble,
        templates=prompt_templates,
    )
    weights = []
    for prompts in prompts_by_class:
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        text_feature = F.normalize(text_features.mean(dim=0), dim=0)
        weights.append(text_feature)
    return torch.stack(weights, dim=1)


@torch.no_grad()
def extract_clip_features(
    model,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
    desc: str = "Extracting CLIP features",
    feature_augmentations: list[str] | None = None,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    features, targets, indices, paths, class_names = [], [], [], [], []
    augmentations = feature_augmentations or ["identity"]
    for batch in tqdm(loader, desc=desc):
        images = batch["image"].to(device)
        feat = encode_image_with_tta(model, images, augmentations)
        features.append(feat.cpu())
        targets.append(batch["target"].cpu())
        indices.extend(batch["index"])
        paths.extend(batch["path"])
        class_names.extend(batch["class_name"])
    return {
        "features": torch.cat(features).numpy(),
        "targets": torch.cat(targets).numpy(),
        "indices": indices,
        "paths": paths,
        "class_names": class_names,
    }


@torch.no_grad()
def encode_image_with_tta(model, images: torch.Tensor, augmentations: list[str]) -> torch.Tensor:
    view_features = []
    for aug in augmentations:
        view = apply_tensor_augmentation(images, aug)
        feat = model.encode_image(view)
        view_features.append(F.normalize(feat, dim=-1))
    feat = torch.stack(view_features, dim=0).mean(dim=0)
    return F.normalize(feat, dim=-1)


def apply_tensor_augmentation(images: torch.Tensor, augmentation: str) -> torch.Tensor:
    if augmentation == "identity":
        return images
    if augmentation == "hflip":
        return torch.flip(images, dims=(-1,))
    if augmentation == "vflip":
        return torch.flip(images, dims=(-2,))
    if augmentation == "rot90":
        return torch.rot90(images, k=1, dims=(-2, -1))
    if augmentation == "rot180":
        return torch.rot90(images, k=2, dims=(-2, -1))
    if augmentation == "rot270":
        return torch.rot90(images, k=3, dims=(-2, -1))
    raise ValueError(f"Unknown CLIP feature augmentation: {augmentation}")


def clip_logits(features: np.ndarray, zeroshot_weights: torch.Tensor) -> np.ndarray:
    weights = zeroshot_weights.detach().cpu().numpy()
    logits = features @ weights
    return logits.astype(np.float64)


def save_logits(logits: np.ndarray, path: str | Path) -> None:
    np.save(Path(path), logits)
