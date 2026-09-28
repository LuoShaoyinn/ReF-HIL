#!/usr/bin/env python3
"""Train the task RGB-D success classifier from tagged PNG pairs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import warnings
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import load_component  # noqa: E402


@dataclass(frozen=True)
class Sample:
    rgb_path: Path
    depth_path: Path
    label: int

    @property
    def group(self) -> str:
        return str(self.rgb_path.parent)


def _resolve_path(value: str, *, labels_path: Path) -> Path:
    path = Path(value)
    candidates = (path, labels_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _paired_depth_path(rgb_path: Path) -> Path:
    name = rgb_path.name
    if not name.endswith("_rgb.png"):
        raise ValueError(f"not a classifier RGB PNG: {rgb_path}")
    prefix = name.removesuffix("_rgb.png")
    candidates = (
        rgb_path.with_name(f"{prefix}_depth_depth.png"),
        rgb_path.with_name(f"{prefix}_depth.png"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no paired depth PNG for {rgb_path}; tried {candidates}")


def read_samples(labels_path: Path, *, allow_incomplete: bool = False) -> list[Sample]:
    """Read append-only labels, retaining the last label for each RGB image."""

    latest: dict[str, tuple[str, str | None]] = {}
    for line_number, line in enumerate(labels_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            image = str(row["image"])
            label = str(row["label"]).lower()
            explicit_depth = None if row.get("depth") is None else str(row["depth"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid label row {labels_path}:{line_number}") from exc
        if label not in {"success", "failure", "skip"}:
            raise ValueError(f"invalid label {label!r} at {labels_path}:{line_number}")
        latest[image] = (label, explicit_depth)

    samples: list[Sample] = []
    for image, (label, explicit_depth) in latest.items():
        if label == "skip":
            continue
        rgb_path = _resolve_path(image, labels_path=labels_path)
        if not rgb_path.is_file():
            if allow_incomplete:
                warnings.warn(f"skipping missing tagged RGB PNG: {rgb_path}")
                continue
            raise FileNotFoundError(f"missing tagged RGB PNG: {rgb_path}")
        try:
            depth_path = (
                _resolve_path(explicit_depth, labels_path=labels_path)
                if explicit_depth is not None
                else _paired_depth_path(rgb_path)
            )
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing paired depth PNG: {depth_path}")
        except FileNotFoundError as exc:
            if not allow_incomplete:
                raise
            warnings.warn(f"skipping incomplete RGB-D pair: {exc}")
            continue
        samples.append(Sample(rgb_path=rgb_path, depth_path=depth_path, label=int(label == "success")))
    if not samples:
        raise ValueError(f"no usable labels found in {labels_path}")
    return samples


def grouped_split(
    samples: list[Sample], *, val_fraction: float, seed: int, allow_incomplete: bool = False
) -> tuple[list[Sample], list[Sample]]:
    """Hold out whole episode directories to avoid adjacent-frame leakage."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError("training requires both success and failure examples")
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        if allow_incomplete:
            warnings.warn("No episode-held-out validation possible; training on all labeled pairs")
            return samples, []
        raise ValueError("need at least two episode directories for a held-out validation split")
    val_group_count = min(len(groups) - 1, max(1, int(round(len(groups) * val_fraction))))
    rng = random.Random(seed)
    for _ in range(1000):
        rng.shuffle(groups)
        validation_groups = set(groups[:val_group_count])
        train = [sample for sample in samples if sample.group not in validation_groups]
        validation = [sample for sample in samples if sample.group in validation_groups]
        if {sample.label for sample in train} == {0, 1} and {sample.label for sample in validation} == {0, 1}:
            return train, validation
    if allow_incomplete:
        warnings.warn("No two-class episode-held-out split possible; training on all labeled pairs")
        return samples, []
    raise ValueError(
        "could not make an episode-held-out split containing both labels in train and validation"
    )


class RGBDPNGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Iterable[Sample],
        *,
        image_size: int,
        normalize_rgbd: object,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.normalize_rgbd = normalize_rgbd

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        rgb_bgr = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(sample.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None:
            raise RuntimeError(f"failed to read RGB PNG: {sample.rgb_path}")
        if depth is None:
            raise RuntimeError(f"failed to read depth PNG: {sample.depth_path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(self.normalize_rgbd(rgb, depth, image_size=self.image_size))
        y = torch.tensor(float(sample.label), dtype=torch.float32)
        return x, y


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
    targets = labels.to(torch.int64)
    positives = targets == 1
    negatives = targets == 0
    accuracy = float((predictions == targets).float().mean().item())
    tpr = float((predictions[positives] == 1).float().mean().item())
    tnr = float((predictions[negatives] == 0).float().mean().item())
    return {"accuracy": accuracy, "balanced_accuracy": 0.5 * (tpr + tnr)}


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    weighted_loss = 0.0
    sample_count = 0
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).reshape(-1)
            batch_size = int(labels.shape[0])
            weighted_loss += float(criterion(logits, labels).item()) * batch_size
            sample_count += batch_size
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    metrics = _metrics(torch.cat(logits_list), torch.cat(labels_list))
    metrics["loss"] = weighted_loss / sample_count
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="assemble", help="package name below tasks/")
    parser.add_argument(
        "--condition",
        default=None,
        help="success-condition name; defaults to the task's first condition",
    )
    parser.add_argument("--labels", type=Path, required=True, help="JSONL from tools/tag_images.py")
    parser.add_argument("--output", type=Path, required=True, help="classifier.pt checkpoint path")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="skip missing RGB-D pairs; train without validation if a grouped split is impossible (both labels still required)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("--epochs and --batch-size must be positive; --num-workers must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classifier_component = load_component(args.task, "classifier")
    task = load_component(args.task, "config").TaskConfig()
    condition = (
        task.success_conditions[0]
        if args.condition is None
        else task.success_condition(args.condition)
    )
    config = classifier_component.ClassifierConfig(
        experiment_name=args.output.parent.name,
        device=args.device,
        pretrained=bool(args.pretrained),
        task=task,
        condition=condition,
    )
    samples = read_samples(args.labels, allow_incomplete=args.allow_incomplete)
    train_samples, validation_samples = grouped_split(
        samples, val_fraction=args.val_fraction, seed=args.seed,
        allow_incomplete=args.allow_incomplete,
    )
    train_labels = np.asarray([sample.label for sample in train_samples], dtype=np.float32)
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    device = torch.device(args.device)
    train_loader = DataLoader(
        RGBDPNGDataset(
            train_samples,
            image_size=condition.image_size,
            normalize_rgbd=classifier_component.normalize_rgbd,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        RGBDPNGDataset(
            validation_samples,
            image_size=condition.image_size,
            normalize_rgbd=classifier_component.normalize_rgbd,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    classifier = classifier_component.Classifier(config=config)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        classifier.model.train()
        weighted_train_loss = 0.0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier.model(images).reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            weighted_train_loss += float(loss.detach().item()) * batch_size
            train_count += batch_size
        if not validation_samples:
            row = {"epoch": epoch, "train_loss": weighted_train_loss / train_count}
            history.append(row)
            print(f"epoch {epoch:03d}/{args.epochs:03d} train_loss={row['train_loss']:.5f} (no validation)")
            best_epoch = epoch
            best_state = copy.deepcopy(classifier.model.state_dict())
            continue
        validation_metrics = _evaluate(classifier.model, validation_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": weighted_train_loss / train_count,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} val_acc={row['val_accuracy']:.3f} "
            f"val_bal_acc={row['val_balanced_accuracy']:.3f}"
        )
        if float(validation_metrics["loss"]) < best_loss:
            best_epoch = epoch
            best_loss = float(validation_metrics["loss"])
            best_state = copy.deepcopy(classifier.model.state_dict())

    assert best_state is not None
    classifier.model.load_state_dict(best_state)
    checkpoint = classifier.save(args.output)
    metrics_path = args.metrics_output or checkpoint.with_name("classifier_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "labels": str(args.labels),
                "task": args.task,
                "condition": condition.name,
                "pretrained": bool(args.pretrained),
                "resnet": config.resnet_name,
                "in_channels": 4,
                "classifier_image_size": int(condition.image_size),
                "classifier_clip": list(condition.clip),
                "normalization": "RGB/depth uint8 -> /255; ImageNet RGB; depth -> [-1,1]",
                "split": "episode-directory-held-out" if validation_samples else "none-training-only",
                "checkpoint_selection": "validation-loss" if validation_samples else "final-epoch",
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "best_epoch": best_epoch,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {'best' if validation_samples else 'final'} epoch {best_epoch} classifier checkpoint: {checkpoint}")
    print(f"saved training metrics: {metrics_path}")


if __name__ == "__main__":
    main()
