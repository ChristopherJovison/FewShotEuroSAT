from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from tqdm import tqdm

from .data import collate_batch


def convnext_transforms(weights_name: str = "IMAGENET1K_V1"):
    weights = ConvNeXt_Tiny_Weights[weights_name]
    return weights.transforms()


def convnext_train_transforms(weights_name: str = "IMAGENET1K_V1"):
    ConvNeXt_Tiny_Weights[weights_name]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.25),
            transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.25),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )


def build_convnext_head(num_classes: int, weights_name: str = "IMAGENET1K_V1") -> nn.Module:
    weights = ConvNeXt_Tiny_Weights[weights_name]
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def train_convnext_head(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mode: str = "linear_probe",
) -> nn.Module:
    if mode not in {"linear_probe", "head_finetune"}:
        raise ValueError("mode must be 'linear_probe' or 'head_finetune'.")
    freeze_backbone(model)
    model.to(device)
    train_features, train_targets, _, _, _ = extract_convnext_features(model, train_dataset, device, batch_size, "ConvNeXt support features")
    val_features, val_targets, _, _, _ = extract_convnext_features(model, val_dataset, device, batch_size, "ConvNeXt val features")
    head = fit_linear_head(train_features, train_targets, val_features, val_targets, device, batch_size, epochs, lr, weight_decay)
    model.classifier[-1].load_state_dict({k: v.detach().cpu() for k, v in head.state_dict().items()})
    model.eval()
    return model


def train_convnext_full(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    epochs: int = 5,
    lr: float = 5e-5,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.05,
) -> tuple[nn.Module, dict[str, object]]:
    model.to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_acc = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Fine-tuning ConvNeXt epoch {epoch + 1}/{epochs}"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            batch_size_seen = int(targets.numel())
            running_loss += float(loss.detach().item()) * batch_size_seen
            seen += batch_size_seen
        scheduler.step()
        val_acc, val_loss = evaluate_convnext_full(model, val_loader, device, criterion)
        train_loss = running_loss / max(seen, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_accuracy": best_val_acc, "history": history}


@torch.no_grad()
def evaluate_convnext_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        logits = model(images)
        if criterion is not None:
            loss_sum += float(criterion(logits, targets).detach().item()) * int(targets.numel())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1), loss_sum / max(total, 1)


@torch.no_grad()
def predict_convnext_full(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    logits, targets, indices, paths, class_names = [], [], [], [], []
    model.eval().to(device)
    for batch in tqdm(loader, desc="ConvNeXt full-finetune test"):
        images = batch["image"].to(device)
        logits.append(model(images).detach().cpu())
        targets.append(batch["target"].cpu())
        indices.extend(batch["index"])
        paths.extend(batch["path"])
        class_names.extend(batch["class_name"])
    return {
        "logits": torch.cat(logits).numpy().astype(np.float64),
        "targets": torch.cat(targets).numpy(),
        "indices": indices,
        "paths": paths,
        "class_names": class_names,
    }


def fit_linear_head(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> nn.Linear:
    in_features = train_features.shape[1]
    num_classes = int(max(train_targets.max().item(), val_targets.max().item())) + 1
    head = nn.Linear(in_features, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_acc = -1.0
    for _ in tqdm(range(epochs), desc="Training ConvNeXt head"):
        head.train()
        order = torch.randperm(train_features.shape[0], device=device)
        for start in range(0, train_features.shape[0], batch_size):
            idx = order[start:start + batch_size]
            features = train_features[idx]
            targets = train_targets[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), targets)
            loss.backward()
            optimizer.step()
        val_acc = _feature_accuracy(head, val_features, val_targets, batch_size)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head


@torch.no_grad()
def predict_with_head(
    head: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: list[int],
    paths: list[str],
    class_names: list[str],
    batch_size: int = 64,
) -> dict[str, object]:
    logits = []
    head.eval()
    for start in range(0, features.shape[0], batch_size):
        logits.append(head(features[start:start + batch_size]).detach().cpu())
    return {
        "logits": torch.cat(logits).numpy().astype(np.float64),
        "targets": targets.detach().cpu().numpy(),
        "indices": indices,
        "paths": paths,
        "class_names": class_names,
    }


@torch.no_grad()
def predict_convnext(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda", collate_fn=collate_batch)
    logits, targets, indices, paths, class_names = [], [], [], [], []
    model.eval().to(device)
    features, target_tensor, indices, paths, class_names = extract_convnext_features(model, dataset, device, batch_size, "ConvNeXt test features")
    head = model.classifier[-1].to(device).eval()
    for start in range(0, features.shape[0], batch_size):
        logits.append(head(features[start:start + batch_size]).detach().cpu())
    return {
        "logits": torch.cat(logits).numpy().astype(np.float64),
        "targets": target_tensor.cpu().numpy(),
        "indices": indices,
        "paths": paths,
        "class_names": class_names,
    }


@torch.no_grad()
def extract_convnext_features(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    desc: str,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str], list[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=collate_batch)
    features, targets, indices, paths, class_names = [], [], [], [], []
    model.eval()
    for batch in tqdm(loader, desc=desc):
        images = batch["image"].to(device)
        feat = model.features(images)
        feat = model.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = F.normalize(feat, dim=1)
        features.append(feat)
        targets.append(batch["target"].to(device))
        indices.extend(batch["index"])
        paths.extend(batch["path"])
        class_names.extend(batch["class_name"])
    return torch.cat(features), torch.cat(targets), indices, paths, class_names


@torch.no_grad()
def _feature_accuracy(head: nn.Module, features: torch.Tensor, targets: torch.Tensor, batch_size: int) -> float:
    correct, total = 0, 0
    head.eval()
    for start in range(0, features.shape[0], batch_size):
        logits = head(features[start:start + batch_size])
        batch_targets = targets[start:start + batch_size]
        correct += int((logits.argmax(dim=1) == batch_targets).sum().item())
        total += int(batch_targets.numel())
    return correct / max(total, 1)
