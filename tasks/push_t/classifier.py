"""Push-T success classifier configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.task.manipulation_classifier import (
    Classifier as BaseClassifier,
    ClassifierConfig as BaseClassifierConfig,
    clip_rgbd,
    normalize_rgbd,
)
from .config import TaskConfig


@dataclass(kw_only=True)
class ClassifierConfig(BaseClassifierConfig):
    task: TaskConfig = field(default_factory=TaskConfig)


class Classifier(BaseClassifier):
    pass


__all__ = ["Classifier", "ClassifierConfig", "clip_rgbd", "normalize_rgbd"]
