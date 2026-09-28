"""Physical and learning configuration for gear insertion."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shared.task.manipulation_config import SuccessCondition, TCPPositionCondition, TaskConfig as BaseTaskConfig


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "insert_gear"
    tcp_success_condition: TCPPositionCondition | None = TCPPositionCondition(
        target=(-0.1180, -0.4600, 0.035), radius=0.010,
    )

    # Policy action: [dx, dy, dz, local drz]. The gripper is held closed by
    # the robot task and is deliberately absent from policy/replay actions.
    action_dims: int = 4
    allow_rotation: bool = True
    movable_axes: tuple[bool, bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        False,
        False,
        True,
    )
    max_episode_steps: int = 150
    step_penalty: float = 1.0 / max_episode_steps

    # Preserve the original workspace offsets around the remapped installed-
    # gear center. The center moved by (+24.5, +74.3, +1.0) mm.
    safety_pos_min: tuple[float, float, float] = (-0.1755, -0.5257, 0.033)
    safety_pos_max: tuple[float, float, float] = (0.0245, -0.3507, 0.121)
    yaw_range_rad: tuple[float, float] = (
        -math.pi / 2.0,
        math.pi / 9.0,
    )
    # Match rotate_knob's responsive local-RZ control without changing this
    # task's calibrated yaw safety range or reset orientation.
    yaw_speed_limit_rad_s: float = math.radians(90)
    angular_speed: float = math.radians(9)
    reset_yaw_waypoint_step_rad: float = math.radians(30)

    # Uniformly sample the complete XY safety workspace at fixed maximum Z.
    reset_pos: tuple[float, float, float] = (-0.0755, -0.4382, 0.121)
    reset_rnd_abs: tuple[float, float, float] = (0.1000, 0.0875, 0.0)
    reset_yaw_rad: float = 0.0
    # Descend at the current XY only when above this release height.
    reset_release_z: float = 0.048
    reset_manual_wait_s: float = 5.0
    # Mechanical settling after each open/close command (not a grasp sensor).
    reset_gripper_wait_s: float = 1.0
    reset_gripper_close_wait_s: float = 2.0
    reset_stage_timeout_s: float = 10.0
    reset_position_tolerance: float = 0.005
    reset_rotation_tolerance: float = math.pi / 180.0
    reset_step_duration_s: float = 0.05
    reset_settle_time_s: float = 1.0

    # The robot-driver startup workflow visits this pose, waits for explicit
    # operator confirmation to close, and then lifts to reset height. Ordinary
    # actor/recorder resets release the gear and automatically regrasp here.
    startup_pickup_pos: tuple[float, float, float] = (-0.1180, -0.4600, 0.035)
    startup_pickup_yaw_rad: float = 0.0
    startup_move_timeout_s: float = 10.0
    fixed_gripper_action: float = 1.0
    gripper_speed: int = 0
    gripper_force: int = 255

    recovery_tilt_half_range_rad: float = math.pi / 6.0
    rotation_recovery_max_wrench: tuple[float, float, float] = (4.0, 4.0, 4.0)

    # Acquire the global camera at 1280x720. Raw crop coordinates use
    # (x, y, width, height). Task success is the logical AND of the two
    # independently trained classifier decisions.
    global_camera_resolution: tuple[int, int] | None = (1280, 720)
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(
            name="classifier_1",
            clip=(685, 299, 80, 82),
            image_key="classifier_classifier_1",
            depth_key="classifier_classifier_1_depth",
            checkpoint="classifiers/classifier_1.pt",
        ),
        SuccessCondition(
            name="classifier_2",
            clip=(760, 304, 67, 67),
            image_key="classifier_classifier_2",
            depth_key="classifier_classifier_2_depth",
            checkpoint="classifiers/classifier_2.pt",
        ),
    )
    global_policy_clip: tuple[int, int, int, int] = (663, 138, 294, 294)
    wrist_0_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (80, 0, 480, 480)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.yaw_speed_limit_rad_s <= 0:
            raise ValueError("yaw speed limit must be positive")
        if self.angular_speed <= 0:
            raise ValueError("angular action step must be positive")
        if self.reset_yaw_waypoint_step_rad <= 0:
            raise ValueError("reset yaw waypoint step must be positive")
