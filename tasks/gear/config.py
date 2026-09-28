"""Physical and learning configuration for the gear task."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shared.task.manipulation_config import SuccessCondition, TaskConfig as BaseTaskConfig


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "gear"

    # Policy action: [dx, dy, dz, local drz, gripper]. Roll and pitch are held
    # at the canonical straight-up orientation by compliant impedance control.
    action_dims: int = 5
    allow_rotation: bool = True
    movable_axes: tuple[bool, bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        False,
        False,
        True,
    )

    # Cartesian limits are expressed in metres. The requested y range is
    # interpreted as [-620, -420] mm, consistent with the -450 mm reset center.
    safety_pos_min: tuple[float, float, float] = (-0.300, -0.620, 0.040)
    safety_pos_max: tuple[float, float, float] = (-0.110, -0.420, 0.150)
    yaw_range_rad: tuple[float, float] = (-math.pi, 0.0)

    reset_pos: tuple[float, float, float] = (-0.200, -0.450, 0.110)
    reset_rnd_abs: tuple[float, float, float] = (0.050, 0.020, 0.0)
    reset_yaw_rad: float = 0.0
    reset_stage_timeout_s: float = 2.0
    reset_position_tolerance: float = 0.005
    reset_rotation_tolerance: float = math.pi / 180.0
    reset_step_duration_s: float = 0.05
    reset_gripper: float = -1.0

    # Permit bounded measured tilt for recovery while normal policy targets
    # always retain canonical roll/pitch.
    recovery_tilt_half_range_rad: float = math.pi / 6.0

    rotation_recovery_max_wrench: tuple[float, float, float] = (4.0, 4.0, 4.0)

    # Crop format is (x, y, width, height). The supplied classifier/global
    # values are opposite corners: [33,142]-[145,254] and
    # [170,10]-[618,458]. Wrist frames are 640x480, so x=80 centers 480x480.
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(33, 142, 112, 112)),
    )
    global_policy_clip: tuple[int, int, int, int] = (170, 10, 448, 448)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
