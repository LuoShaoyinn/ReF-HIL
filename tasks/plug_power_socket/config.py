"""Physical and visual configuration for plugging a power socket."""

from __future__ import annotations

from dataclasses import dataclass

from shared.task.manipulation_config import (
    SuccessCondition,
    TaskConfig as BaseTaskConfig,
)


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "plug_power_socket"

    # Policy action: [dx, dy, dz, gripper].
    action_dims: int = 4
    allow_rotation: bool = False
    max_episode_steps: int = 200
    step_penalty: float = 1.0 / max_episode_steps
    gripper_speed: int = 128
    gripper_force: int = 255

    # Cartesian workspace in metres. Rotation is not policy-controlled; retain
    # the shared upright pose at zero local tool yaw.
    safety_pos_min: tuple[float, float, float] = (-0.130, -0.550, -0.005)
    safety_pos_max: tuple[float, float, float] = (0.000, -0.450, 0.030)
    zero_point_rot: tuple[float, float, float] = (2.221, 2.221, 0.0)

    # Reset at the maximum X/Z boundary and sample the complete Y range.
    reset_pos: tuple[float, float, float] = (0.000, -0.500, 0.030)
    reset_rnd_abs: tuple[float, float, float] = (0.0, 0.050, 0.0)
    reset_rot_rnd_abs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reset_settle_time_s: float = 3.0

    # Literal crop coordinates calibrated in this task's 1280x720 global
    # RGB-D stream. Wrist cameras remain at their shared 640x480 profile.
    global_camera_resolution: tuple[int, int] | None = (1280, 720)
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(695, 255, 78, 78)),
    )
    global_policy_clip: tuple[int, int, int, int] = (647, 202, 368, 358)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
