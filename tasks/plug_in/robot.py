"""Plug-in robot component using shared Cartesian manipulation control."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.task.manipulation_robot import (
    Robot as BaseRobot,
    RobotConfig as BaseRobotConfig,
    _crop,
)
from .config import TaskConfig


@dataclass(kw_only=True)
class RobotConfig(BaseRobotConfig):
    task: TaskConfig = field(default_factory=TaskConfig)


class Robot(BaseRobot):
    pass


__all__ = ["Robot", "RobotConfig", "_crop"]
