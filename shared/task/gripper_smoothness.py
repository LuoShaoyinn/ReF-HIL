"""Episode-local gripper-direction history and graded success reward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class GripperSmoothnessConfig:
    direction_deadband: float = 0.1
    reversal_penalty: float = 0.1
    success_reward_floor: float = 0.5

class GripperSmoothnessTracker:
    """Count meaningful changes in commanded gripper movement direction.

    Small command changes accumulate against the last significant command,
    rather than being discarded one sample at a time.
    """

    def __init__(
        self,
        config: GripperSmoothnessConfig,
        *,
        initial_command: float = -1.0,
    ) -> None:
        self.config = config
        self.reset(initial_command=initial_command)

    def reset(self, *, initial_command: float = -1.0) -> None:
        self._anchor_command = float(np.clip(initial_command, -1.0, 1.0))
        self.direction = 0
        self.reversals = 0

    def observe(self, command: float) -> bool:
        command = float(np.clip(command, -1.0, 1.0))
        delta = command - self._anchor_command
        if abs(delta) < self.config.direction_deadband:
            return False
        new_direction = 1 if delta > 0.0 else -1
        reversed_direction = self.direction != 0 and new_direction != self.direction
        if reversed_direction:
            self.reversals += 1
        self.direction = new_direction
        self._anchor_command = command
        return reversed_direction

    @property
    def success_reward(self) -> float:
        return max(
            self.config.success_reward_floor,
            1.0 - self.config.reversal_penalty * self.reversals,
        )

def tracker_for_task(task: object, *, initial_command: float = -1.0) -> GripperSmoothnessTracker:
    """Construct a tracker from the shared task reward configuration."""

    return GripperSmoothnessTracker(
        GripperSmoothnessConfig(
            direction_deadband=float(
                getattr(task, "gripper_direction_deadband", 0.1)
            ),
            reversal_penalty=float(
                getattr(task, "gripper_reversal_penalty", 0.1)
            ),
            success_reward_floor=float(
                getattr(task, "success_reward_floor", 0.5)
            ),
        ),
        initial_command=initial_command,
    )
