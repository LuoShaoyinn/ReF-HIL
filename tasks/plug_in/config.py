"""Physical and visual configuration for the plug-in task."""

from __future__ import annotations

from dataclasses import dataclass

from shared.task.manipulation_config import (
    SuccessCondition,
    TaskConfig as BaseTaskConfig,
)


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "plug_in"

    # Policy action: [dx, dy, dz, gripper].
    action_dims: int = 4
    allow_rotation: bool = False
    max_episode_steps: int = 200
    step_penalty: float = 1.0 / max_episode_steps
    # Close slowly but retain maximum available grasp force.
    gripper_speed: int = 0
    gripper_force: int = 255

    # Supplied millimetre bounds converted to metres.
    safety_pos_min: tuple[float, float, float] = (-0.250, -0.720, 0.030)
    safety_pos_max: tuple[float, float, float] = (-0.150, -0.470, 0.110)
    # Sample x/y across the workspace while keeping reset height at 100 mm.
    reset_pos: tuple[float, float, float] = (-0.200, -0.595, 0.100)
    reset_rnd_abs: tuple[float, float, float] = (0.050, 0.125, 0.0)
    reset_rot_rnd_abs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reset_settle_time_s: float = 3.0

    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(292, 108, 112, 112)),
    )
    global_policy_clip: tuple[int, int, int, int] = (158, 86, 224, 224)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
