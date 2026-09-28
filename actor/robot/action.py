from __future__ import annotations

import numpy as np
from typing import Protocol


class PositionGripper(Protocol):
    def get_open_position(self) -> int: ...
    def get_closed_position(self) -> int: ...
    def move(self, position: int, speed: int, force: int) -> object: ...


# Robotiq SPE uses [0, 255]. 64 is the same low speed used by calibration.
GRIPPER_COMMAND_SPEED = 64
GRIPPER_COMMAND_FORCE = 0


def normalized_gripper_position(
    command: float,
    open_position: int,
    closed_position: int,
) -> int:
    """Map a continuous [-1, 1] action to a calibrated gripper position."""

    normalized = float(np.clip(command, -1.0, 1.0))
    return int(round(
        int(open_position)
        + 0.5 * (normalized + 1.0)
        * (int(closed_position) - int(open_position))
    ))


def send_continuous_gripper_action(
    gripper: PositionGripper,
    command: float,
    *,
    speed: int = GRIPPER_COMMAND_SPEED,
    force: int = GRIPPER_COMMAND_FORCE,
) -> int:
    """Send a continuous target with task-selected Robotiq parameters."""

    position = normalized_gripper_position(
        command,
        gripper.get_open_position(),
        gripper.get_closed_position(),
    )
    gripper.move(position, int(speed), int(force))
    return position
