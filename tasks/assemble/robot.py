from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.task.manipulation_robot import Robot as BaseRobot
from shared.task.manipulation_robot import RobotConfig as BaseRobotConfig
from .config import TaskConfig


def normalized_gripper_position(
    value: float,
    minimum: int,
    maximum: int,
    openness_range: tuple[float, float],
) -> int:
    value = float(np.clip(value, -1.0, 1.0))
    openness_min, openness_max = openness_range
    if not 0.0 <= openness_min < openness_max <= 1.0:
        raise ValueError(f"invalid gripper openness range {openness_range}")
    action_fraction = 0.5 * (value + 1.0)
    openness = openness_max - action_fraction * (openness_max - openness_min)
    closed_fraction = 1.0 - openness
    return int(round(float(minimum) + closed_fraction * float(maximum - minimum)))


@dataclass(kw_only=True)
class RobotConfig(BaseRobotConfig):
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.impendence_control.rotation_recovery = None


class Robot(BaseRobot):
    config: RobotConfig

    def gripper_action_range(self) -> tuple[int, int]:
        # Action +1 closes only through the task's allowed partial travel.
        return tuple(
            normalized_gripper_position(
                command, self.gripper.get_open_position(),
                self.gripper.get_closed_position(),
                self.config.task.gripper_openness_range,
            )
            for command in (-1.0, 1.0)
        )

    def apply_action(self, raw_action: dict) -> None:
        movement = dict(raw_action)
        movement.pop("gripper", None)
        super().apply_action(movement)
        if "gripper" in raw_action:
            task = self.config.task
            position = normalized_gripper_position(
                float(raw_action["gripper"]),
                self.gripper.get_open_position(),
                self.gripper.get_closed_position(),
                task.gripper_openness_range,
            )
            self.gripper.move(position, task.gripper_speed, task.gripper_force)
