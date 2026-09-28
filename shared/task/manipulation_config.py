"""Shared defaults for camera-based Cartesian manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, kw_only=True)
class SuccessCondition:
    """One independently trained RGB-D condition in the task success AND."""

    name: str
    clip: tuple[int, int, int, int]
    image_size: int = 224
    threshold: float = 0.5
    # Preserve the historical contract for the first/single classifier while
    # allowing additional conditions to use distinct observation keys/files.
    image_key: str = "classifier"
    depth_key: str = "classifier_depth"
    checkpoint: str = "classifier.pt"


@dataclass(frozen=True, kw_only=True)
class TCPPositionCondition:
    """Strict Euclidean position-distance gate, in metres; no orientation test."""

    target: tuple[float, float, float]
    radius: float

    def __post_init__(self):
        if len(self.target) != 3 or not all(math.isfinite(v) for v in self.target):
            raise ValueError("TCP success target must contain three finite coordinates")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("TCP success radius must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class TaskConfig:
    name: str = "manipulation"
    tcp_success_condition: TCPPositionCondition | None = None

    # Model-facing geometry.
    obs_dims: int = 1552
    action_dims: int = 4
    mechanism_obs_dims: int = 16
    max_episode_steps: int = 100
    # Positive magnitude of the reward charged on every non-success step.
    # Actor/recorder store its negative; learner horizon constraints use the
    # same positive magnitude.
    step_penalty: float = 0.01
    gripper_direction_deadband: float = 0.1
    gripper_reversal_penalty: float = 0.1
    success_reward_floor: float = 0.5
    image_embedding_dims: int = 512
    image_keys: tuple[str, ...] = ("global_0", "wrist_0", "wrist_1")

    # Deterministic observation normalization.
    tcp_speed_range: tuple[tuple[float, ...], tuple[float, ...]] = (
        (-0.05, -0.05, -0.05, -0.4, -0.4, -0.4),
        (0.05, 0.05, 0.05, 0.4, 0.4, 0.4),
    )
    tcp_force_range: tuple[tuple[float, ...], tuple[float, ...]] = (
        (-40.0, -40.0, -40.0, -1.0, -1.0, -1.0),
        (40.0, 40.0, 40.0, 1.0, 1.0, 1.0),
    )
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Normalized action -> physical robot motion.
    linear_speed: float = 0.05
    angular_speed: float = 0.10
    allow_rotation: bool = False
    # Cartesian controller mode in XYZ/RX/RY/RZ order. Movable axes retain
    # the normal compliant PID; fixed axes use strong-far/soft-near holding.
    movable_axes: tuple[bool, bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        False,
        False,
        False,
    )

    # Task reset trajectory.
    reset_pos: tuple[float, float, float] = (-0.250, -0.580, 0.060)
    reset_rnd_abs: tuple[float, float, float] = (0.040, 0.040, 0.005)
    reset_lift_delta: tuple[float, float, float] = (0.0, 0.0, 0.3)
    reset_lift_duration_s: float = 0.5
    reset_step_duration_s: float = 0.1
    reset_position_tolerance: float = 0.01
    reset_rot_rnd_abs: tuple[float, float, float] = (
        math.pi / 36.0,
        math.pi / 36.0,
        math.pi / 36.0,
    )
    reset_rotation_tolerance: float = math.pi / 180.0
    reset_move_timeout_s: float = 2.0
    # Optional pause after reset motion completes and before an episode may
    # start. This is distinct from reset_move_timeout_s, which only bounds the
    # active move-to loop.
    reset_settle_time_s: float = 0.0
    reset_gripper: float = -1.0
    gripper_position_range: tuple[float, float] = (0.0, 255.0)
    gripper_speed: int = 64
    gripper_force: int = 0

    # Task workspace and nominal orientation.
    zero_point_rot: tuple[float, float, float] = (2.221, 2.221, 0.0)
    safety_pos_min: tuple[float, float, float] = (-0.300, -0.750, 0.005)
    safety_pos_max: tuple[float, float, float] = (-0.200, -0.530, 0.150)
    # Total 60-degree window on each local rotation axis: -30 to +30 degrees.
    safety_rot_half_range: tuple[float, float, float] = (
        math.pi / 6.0,
        math.pi / 6.0,
        math.pi / 6.0,
    )
    # Once measured pose crosses fence + delta, recover toward this far inside.
    # Recovery doubles rotational gain, raises the bounded torque limit from
    # 2 Nm to 4 Nm, and temporarily disables translation and gripper control.
    rotation_recovery_inward_margin: tuple[float, float, float] = (
        math.pi / 36.0,
        math.pi / 36.0,
        math.pi / 36.0,
    )
    # Strong pull-back starts only after measured orientation exceeds the
    # normal fence by this delta. Commands are still projected onto the fence.
    rotation_recovery_activation_delta: tuple[float, float, float] = (
        math.pi / 36.0,
        math.pi / 36.0,
        math.pi / 36.0,
    )
    rotation_recovery_max_wrench: tuple[float, float, float] = (4.0, 4.0, 4.0)
    rotation_recovery_wrench_gain_multiplier: float = 2.0

    # Human-control copilot context. The SpaceMouse driver's
    # --override-rotation flag activates it.
    # None means use zero_point_rot, keeping copilot and safety center aligned.
    spacemouse_copilot_target_rot: tuple[float, float, float] | None = None
    spacemouse_copilot_gain: float = 1.0
    spacemouse_copilot_deadband_rad: float = 0.01

    # Raw-camera crop format is always (x, y, width, height).
    # None preserves the shared 640x480 RealSense profile. Tasks may request
    # another global RGB-D profile while keeping wrist cameras unchanged.
    global_camera_resolution: tuple[int, int] | None = None
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(name="success", clip=(101, 201, 112, 112)),
    )
    global_policy_clip: tuple[int, int, int, int] = (100, 100, 256, 256)
    wrist_0_policy_clip: tuple[int, int, int, int] = (0, 0, 480, 480)
    wrist_1_policy_clip: tuple[int, int, int, int] = (0, 0, 480, 480)
    policy_image_size: int = 128

    def __post_init__(self) -> None:
        if self.reset_settle_time_s < 0.0:
            raise ValueError("reset_settle_time_s must be non-negative")
        if not 0 <= self.gripper_speed <= 255:
            raise ValueError("gripper_speed must be within [0, 255]")
        if not 0 <= self.gripper_force <= 255:
            raise ValueError("gripper_force must be within [0, 255]")
        if not 0.0 < self.gripper_direction_deadband <= 2.0:
            raise ValueError("gripper_direction_deadband must be within (0, 2]")
        if not 0.0 < self.gripper_reversal_penalty <= 0.5:
            raise ValueError("gripper_reversal_penalty must be within (0, 0.5]")
        if not 0.0 < self.success_reward_floor <= 1.0:
            raise ValueError("success_reward_floor must be within (0, 1]")
        if not self.success_conditions:
            raise ValueError("a task must define at least one success condition")
        if self.global_camera_resolution is not None:
            width, height = self.global_camera_resolution
            if width <= 0 or height <= 0:
                raise ValueError("global_camera_resolution must be positive")
        names: set[str] = set()
        observation_keys: set[str] = set()
        checkpoints: set[str] = set()
        for condition in self.success_conditions:
            if not condition.name or condition.name in names:
                raise ValueError(f"duplicate or empty success-condition name: {condition.name!r}")
            if not 0.0 <= condition.threshold <= 1.0:
                raise ValueError(f"invalid threshold for {condition.name!r}: {condition.threshold}")
            if condition.image_size <= 0:
                raise ValueError(f"invalid image size for {condition.name!r}: {condition.image_size}")
            _, _, width, height = condition.clip
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid crop for {condition.name!r}: {condition.clip}")
            if (
                condition.image_key == condition.depth_key
                or condition.image_key in observation_keys
                or condition.depth_key in observation_keys
            ):
                raise ValueError(f"duplicate success-condition image keys for {condition.name!r}")
            if condition.checkpoint in checkpoints:
                raise ValueError(f"duplicate success-condition checkpoint: {condition.checkpoint!r}")
            names.add(condition.name)
            observation_keys.add(condition.image_key)
            observation_keys.add(condition.depth_key)
            checkpoints.add(condition.checkpoint)

    def success_condition(self, name: str) -> SuccessCondition:
        for condition in self.success_conditions:
            if condition.name == name:
                return condition
        available = ", ".join(condition.name for condition in self.success_conditions)
        raise ValueError(f"unknown success condition {name!r}; available: {available}")
