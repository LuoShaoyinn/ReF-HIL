from __future__ import annotations

from dataclasses import dataclass

from shared.task.manipulation_config import SuccessCondition, TaskConfig as BaseTaskConfig


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "assemble"
    allow_rotation: bool = False
    action_dims: int = 4
    max_episode_steps: int = 200
    # 200 non-success transitions produce an undiscounted timeout return -1.
    step_penalty: float = 1.0 / 200.0

    reset_pos: tuple[float, float, float] = (-0.165, -0.475, 0.100)
    reset_rnd_abs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    safety_pos_min: tuple[float, float, float] = (-0.250, -0.650, 0.005)
    safety_pos_max: tuple[float, float, float] = (-0.080, -0.400, 0.120)

    # Crop format is x, y, width, height.
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(168, 240, 112, 112)),
    )
    global_policy_clip: tuple[int, int, int, int] = (96, 16, 448, 448)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)

    # Policy action -1 is fully open and +1 reaches only 25% of calibrated
    # closing travel. The observation uses the corresponding nominal SDK span.
    gripper_openness_range: tuple[float, float] = (0.75, 1.0)
    gripper_position_range: tuple[float, float] = (0.0, 64.0)
    gripper_speed: int = 64
    gripper_force: int = 0
