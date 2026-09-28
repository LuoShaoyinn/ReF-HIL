"""Knob task profile. Scene-dependent defaults MUST be calibrated before use."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shared.task.manipulation_config import SuccessCondition, TaskConfig as BaseTaskConfig


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "rotate_knob"
    # Four Cartesian DoFs plus one continuous gripper action.
    action_dims: int = 5
    allow_rotation: bool = True
    movable_axes: tuple[bool, bool, bool, bool, bool, bool] = (
        True, True, True, False, False, True,
    )
    max_episode_steps: int = 200

    # Match the canonical double-string task's continuous gripper contract.
    gripper_speed: int = 128
    gripper_force: int = 200
    gripper_position_range: tuple[float, float] = (0.0, 255.0)
    reset_gripper: float = -1.0

    # Calibrated workspace around the knob target (-124, -526, 109) mm. The
    # target lies on the lower Z boundary.
    safety_pos_min: tuple[float, float, float] = (-0.204, -0.601, 0.109)
    safety_pos_max: tuple[float, float, float] = (-0.044, -0.451, 0.160)
    # Reset uniformly over the full XY workspace while holding maximum Z.
    reset_pos: tuple[float, float, float] = (-0.124, -0.526, 0.160)
    reset_rnd_abs: tuple[float, float, float] = (0.080, 0.075, 0.0)
    reset_clearance_z: float = 0.160
    # User-selected local tool-Z range relative to zero_point_rot, not rotvec[2].
    yaw_range_rad: tuple[float, float] = (math.radians(-100), math.radians(100))
    reset_yaw_rad: float = math.radians(100)
    yaw_speed_limit_rad_s: float = math.radians(90)
    # Action scale per 10 Hz control step: 9 deg/step = 90 deg/s.
    # Keep this separate from the impedance controller's rad/s speed limit so
    # repeated commands cannot queue a large rotation target ahead of the arm.
    angular_speed: float = math.radians(9)
    # Reset needs coarse waypoints to force the intended increasing-yaw path
    # without inheriting the small online action increment.
    reset_yaw_waypoint_step_rad: float = math.radians(30)
    recovery_tilt_half_range_rad: float = math.pi / 6
    reset_stage_timeout_s: float = 5.0
    reset_position_tolerance: float = 0.005
    reset_rotation_tolerance: float = math.pi / 180
    reset_step_duration_s: float = 0.05
    human_reset_seconds: int = 5
    reset_total_seconds: float = 7.0

    # Acquire the global camera at native 1280x720. Crop coordinates below use
    # (x, y, width, height) on that native frame.
    global_camera_resolution: tuple[int, int] | None = (1280, 720)
    # Global policy crop covers the panel and gripper approach area.
    # Classifier ROI surrounds the knob and its position marker, not an indicator.
    # Relabel knob success states and train a new classifier.pt for this crop.
    global_policy_clip: tuple[int, int, int, int] = (618, 1, 300, 300)
    wrist_0_policy_clip: tuple[int, int, int, int] = (0, 0, 448, 448)
    wrist_1_policy_clip: tuple[int, int, int, int] = (0, 0, 448, 448)
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(700, 110, 100, 100)),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        lo, hi = self.yaw_range_rad
        if not (-math.pi < lo < hi < math.pi):
            raise ValueError("yaw range must be ordered and strictly inside (-pi, pi)")
        if not lo <= self.reset_yaw_rad <= hi:
            raise ValueError("reset yaw must lie inside yaw_range_rad")
        if not math.isfinite(self.yaw_speed_limit_rad_s) or self.yaw_speed_limit_rad_s <= 0:
            raise ValueError("yaw speed limit must be finite and positive")
        if not math.isfinite(self.angular_speed) or self.angular_speed <= 0:
            raise ValueError("angular action step must be finite and positive")
        if not math.isfinite(self.reset_yaw_waypoint_step_rad) or self.reset_yaw_waypoint_step_rad <= 0:
            raise ValueError("reset yaw waypoint step must be finite and positive")
        if self.reset_stage_timeout_s <= 0 or self.reset_step_duration_s <= 0:
            raise ValueError("reset durations must be positive")
        if self.human_reset_seconds < 0:
            raise ValueError("human_reset_seconds must be nonnegative")
        if not math.isfinite(self.reset_total_seconds) or self.reset_total_seconds < self.human_reset_seconds:
            raise ValueError("reset_total_seconds must be finite and cover human reset time")
        if not self.safety_pos_min[2] <= self.reset_clearance_z <= self.safety_pos_max[2]:
            raise ValueError("reset clearance must lie inside workspace")
        boundary_tolerance = 1e-9
        for low, high, position, spread in zip(
            self.safety_pos_min, self.safety_pos_max, self.reset_pos, self.reset_rnd_abs
        ):
            reset_low = position - spread
            reset_high = position + spread
            if (
                not low < high
                or spread < 0
                or reset_low < low - boundary_tolerance
                or reset_high > high + boundary_tolerance
            ):
                raise ValueError("reset range must lie inside the ordered workspace")
