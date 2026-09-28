"""Physical and learning configuration for the XYZ Push-T task."""

from __future__ import annotations

from dataclasses import dataclass

from shared.task.manipulation_config import SuccessCondition, TaskConfig as BaseTaskConfig


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "push_t"
    # The actor runs at 10 Hz, so 150 steps gives a 15-second episode.
    max_episode_steps: int = 150
    # Preserve the task-level contract that a full timeout sums to -1.
    step_penalty: float = 1.0 / max_episode_steps

    # Policy action: [dx, dy, dz]. The physical gripper is closed by
    # the robot task and is deliberately absent from the learned action.
    action_dims: int = 3
    allow_rotation: bool = False
    movable_axes: tuple[bool, bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        False,
        False,
        False,
    )

    # Cartesian coordinates use metres throughout the runtime. XYZ are all
    # policy-controlled; every rotational axis is held at zero_point_rot.
    safety_pos_min: tuple[float, float, float] = (-0.300, -0.630, 0.025)
    safety_pos_max: tuple[float, float, float] = (-0.080, -0.400, 0.060)

    # Reset only positions the robot; resetting the pushed object remains an
    # operator/environment responsibility. Every reset starts at y=-400 mm
    # and z=60 mm; only x is sampled over the configured workspace.
    reset_pos: tuple[float, float, float] = (-0.190, -0.400, 0.060)
    reset_rnd_abs: tuple[float, float, float] = (0.110, 0.0, 0.0)
    reset_lift_z: float = 0.060
    reset_target_move_timeout_s: float = 2.0
    reset_move_timeout_s: float = 8.0
    reset_position_tolerance: float = 0.002
    # Pause after the robot reaches its reset pose so the operator can place
    # the T-shaped object before the next episode starts.
    reset_settle_time_s: float = 2.0

    # Physical gripper command injected by the task robot and raw-action
    # adapter. It is not a policy dimension.
    fixed_gripper_action: float = 1.0

    # The global D405 uses 1280x720 RGB-D at 30 Hz; wrist streams remain at
    # 640x480. These are the latest interactively selected global-camera
    # crops.
    global_camera_resolution: tuple[int, int] | None = (1280, 720)
    # Crop format is (x, y, width, height).
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(296, 300, 228, 228)),
    )
    global_policy_clip: tuple[int, int, int, int] = (300, 136, 640, 400)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
