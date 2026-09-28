"""Physical and visual configuration for RJ45 insertion."""

from __future__ import annotations

from dataclasses import dataclass

from shared.task.manipulation_config import (
    SuccessCondition,
    TaskConfig as BaseTaskConfig,
)


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "insert_rj45"

    # Deliberately identical to insert_usb except for task identity.
    action_dims: int = 4
    allow_rotation: bool = False
    max_episode_steps: int = 200
    step_penalty: float = 1.0 / max_episode_steps
    gripper_speed: int = 255
    gripper_force: int = 255

    safety_pos_min: tuple[float, float, float] = (-0.175, -0.700, -0.007)
    safety_pos_max: tuple[float, float, float] = (-0.040, -0.425, 0.100)
    reset_pos: tuple[float, float, float] = (-0.150, -0.475, 0.040)
    reset_rnd_abs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reset_rot_rnd_abs: tuple[float, float, float] = (0.0, 0.0, 0.0)

    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(261, 200, 56, 56)),
    )

    global_policy_clip: tuple[int, int, int, int] = (96, 16, 448, 448)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
