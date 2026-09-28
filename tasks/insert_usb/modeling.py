"""USB observation, action, and reset contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.task.manipulation_modeling import (
    Modeling as BaseModeling,
    ModelingConfig as BaseModelingConfig,
)
from .config import TaskConfig


@dataclass(kw_only=True)
class ModelingConfig(BaseModelingConfig):
    task: TaskConfig = field(default_factory=TaskConfig)


class Modeling(BaseModeling):
    pass


__all__ = ["Modeling", "ModelingConfig"]
