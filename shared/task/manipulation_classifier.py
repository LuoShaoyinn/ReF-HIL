"""Shared RGB-D success classifier and crop preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models

from actor.success import ResNetClassifier, ResNetClassifierConfig
from shared.zmq import DictMessage
from .manipulation_config import SuccessCondition, TaskConfig


RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def clip_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    clip: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if clip is None:
        return rgb_array, depth_array
    x, y, width, height = (int(value) for value in clip)
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise ValueError(f"invalid clip {clip}")
    if rgb_array.ndim != 3 or depth_array.ndim != 2:
        raise ValueError("clip expects HxWx3 RGB and HxW depth arrays")
    if rgb_array.shape[:2] != depth_array.shape:
        raise ValueError(f"RGB/depth size mismatch: {rgb_array.shape} vs {depth_array.shape}")
    if x + width > rgb_array.shape[1] or y + height > rgb_array.shape[0]:
        raise ValueError(f"clip {clip} is outside image shape {rgb_array.shape[:2]}")
    return rgb_array[y : y + height, x : x + width], depth_array[y : y + height, x : x + width]


def normalize_rgbd(rgb: np.ndarray, depth: np.ndarray, *, image_size: int) -> np.ndarray:
    rgb_array = np.asarray(rgb, dtype=np.float32)
    depth_array = np.asarray(depth, dtype=np.float32)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {rgb_array.shape}")
    if depth_array.ndim == 3 and depth_array.shape[2] == 1:
        depth_array = depth_array[:, :, 0]
    if depth_array.ndim != 2 or rgb_array.shape[:2] != depth_array.shape:
        raise ValueError(f"RGB/depth size mismatch: {rgb_array.shape} vs {depth_array.shape}")
    if rgb_array.shape[:2] != (image_size, image_size):
        rgb_array = cv2.resize(rgb_array, (image_size, image_size), interpolation=cv2.INTER_AREA)
        depth_array = cv2.resize(depth_array, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    rgb_chw = np.transpose(rgb_array / 255.0, (2, 0, 1))
    rgb_chw = (rgb_chw - RGB_MEAN) / RGB_STD
    depth_chw = (((depth_array / 255.0) - 0.5) / 0.5)[None, :, :]
    return np.concatenate((rgb_chw, depth_chw), axis=0).astype(np.float32, copy=False)


@dataclass(kw_only=True)
class ClassifierConfig(ResNetClassifierConfig):
    task: TaskConfig = field(default_factory=TaskConfig)
    condition: SuccessCondition | None = None
    in_channels: int = 4

    def __post_init__(self) -> None:
        if self.condition is None:
            self.condition = self.task.success_conditions[0]
        self.success_threshold = float(self.condition.threshold)


class Classifier(ResNetClassifier):
    config: ClassifierConfig

    @property
    def checkpoint_path(self) -> Path:
        condition = self.config.condition
        assert condition is not None
        return Path("outputs") / str(self.config.experiment_name) / condition.checkpoint

    def preprocessing_contract(self) -> dict[str, object]:
        condition = self.config.condition
        assert condition is not None
        contract: dict[str, object] = {
            "resnet_name": str(self.config.resnet_name),
            "in_channels": int(self.config.in_channels),
            "classifier_image_size": int(condition.image_size),
            "classifier_clip": tuple(int(value) for value in condition.clip),
        }
        # The migrated primary condition intentionally preserves the exact
        # preprocessing metadata of existing single-classifier checkpoints.
        if condition.name != "success":
            contract["condition_name"] = condition.name
        return contract

    def build_model(self) -> nn.Module:
        model_factory = getattr(tv_models, str(self.config.resnet_name), None)
        if model_factory is None:
            raise ValueError(f"unsupported torchvision resnet model: {self.config.resnet_name}")
        weights = _resnet_weights(str(self.config.resnet_name)) if self.config.pretrained else None
        model = model_factory(weights=weights)
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            new_conv.weight[:, 3:4].copy_(old_conv.weight.mean(dim=1, keepdim=True))
        model.conv1 = new_conv
        model.fc = nn.Linear(int(model.fc.in_features), 1)
        return model

    def modeling(self, raw_robot_observation: DictMessage) -> np.ndarray:
        condition = self.config.condition
        assert condition is not None
        images = raw_robot_observation["images"]
        rgb = np.asarray(images[condition.image_key])
        depth = np.asarray(images[condition.depth_key])
        _, _, width, height = condition.clip
        expected = (height, width)
        if rgb.shape[:2] != expected or depth.shape[:2] != expected:
            raise ValueError(
                f"runtime classifier {condition.name!r} RGB-D must use clip {condition.clip}; "
                f"got RGB {rgb.shape} and depth {depth.shape}"
            )
        return normalize_rgbd(rgb, depth, image_size=condition.image_size)


def _resnet_weights(model_name: str) -> object | None:
    if model_name == "resnet18":
        return tv_models.ResNet18_Weights.IMAGENET1K_V1
    if model_name == "resnet34":
        return tv_models.ResNet34_Weights.IMAGENET1K_V1
    if model_name == "resnet50":
        return tv_models.ResNet50_Weights.IMAGENET1K_V2
    return None
