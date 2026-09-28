import time
from dataclasses import dataclass, field
import numpy as np
import rtde_receive # type: ignore[import]
import rtde_control # type: ignore[import]
from numpy.typing import NDArray
from typing import Annotated, Literal
from scipy.spatial.transform import Rotation as R

# define types
Array2x3 = Annotated[NDArray[np.float32], Literal[2, 3]]
Array3 = Annotated[NDArray[np.float32], Literal[3, ]]


@dataclass(frozen=True, kw_only=True)
class RotationRecoveryConfig:
    """Task-provided orientation fence enforced by the 500 Hz controller."""

    nominal_rotvec: tuple[float, float, float]
    half_range_rotvec: tuple[float, float, float]
    inward_margin_rotvec: tuple[float, float, float]
    activation_delta_rotvec: tuple[float, float, float]
    max_wrench: tuple[float, float, float]
    wrench_gain_multiplier: float = 2.0


@dataclass(kw_only=True)
class URImpedanceControlConfig:
    ip: str = "192.168.0.10"
    port: int = 30004
    freq: float = 500.0
    # Cascaded Cartesian controller, with rows [translation, rotation].
    # Outer loop: pose error -> desired TCP linear/angular velocity.
    # Inner loop: velocity error -> commanded force/torque.
    cascade_gain: dict[str, Array2x3 | Array3] = field(
        default_factory=lambda: {
            "pose_to_speed": np.array(
                [[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]], dtype=np.float32
            ),
            "speed_to_wrench": np.array(
                [[1200.0, 1200.0, 1200.0], [10.0, 10.0, 40.0]], dtype=np.float32
            ),
        }
    )

    # Continuous component-wise pose deadband. Errors inside the band map to
    # zero; outside, only the excess is controlled. Defaults preserve the
    # original controller exactly.
    pose_error_deadzone: Array2x3 = field(
        default_factory=lambda: np.zeros((2, 3), dtype=np.float32)
    )

    # Optional near-target gain scheduling. A positive half-width selects the
    # corresponding soft gains and wrench cap while that pose-error component
    # is inside the zone. Zero disables scheduling for that component.
    soft_pose_zone: Array2x3 = field(
        default_factory=lambda: np.zeros((2, 3), dtype=np.float32)
    )
    soft_pose_to_speed_gain: Array2x3 = field(
        default_factory=lambda: np.array(
            [[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]], dtype=np.float32
        )
    )
    soft_speed_to_wrench_gain: Array2x3 = field(
        default_factory=lambda: np.array(
            [[1200.0, 1200.0, 1200.0], [10.0, 10.0, 40.0]],
            dtype=np.float32,
        )
    )
    soft_max_wrench: Array2x3 = field(
        default_factory=lambda: np.array(
            [[30.0, 30.0, 30.0], [2.0, 2.0, 2.0]], dtype=np.float32
        )
    )

    # Fixed Cartesian axes use a stronger far-field controller, then fall
    # back to the ordinary compliant gains above near their target. Rows are
    # [translation, rotation]. Translation keeps the existing force authority;
    # rotation uses the bounded 30/20/4-Nm profile validated on Gear.
    fixed_axis_pose_to_speed_gain: Array2x3 = field(
        default_factory=lambda: np.full((2, 3), 30.0, dtype=np.float32)
    )
    fixed_axis_speed_to_wrench_gain: Array2x3 = field(
        default_factory=lambda: np.array(
            [[1200.0, 1200.0, 1200.0], [20.0, 20.0, 20.0]],
            dtype=np.float32,
        )
    )
    fixed_axis_max_wrench: Array2x3 = field(
        default_factory=lambda: np.array(
            [[30.0, 30.0, 30.0], [4.0, 4.0, 4.0]],
            dtype=np.float32,
        )
    )
    fixed_axis_soft_zone: Array2x3 = field(
        default_factory=lambda: np.array(
            [[0.005, 0.005, 0.005], [np.deg2rad(5.0)] * 3],
            dtype=np.float32,
        )
    )
    fixed_axis_near_pose_to_speed_gain: Array2x3 = field(
        default_factory=lambda: np.full((2, 3), 20.0, dtype=np.float32)
    )
    fixed_axis_near_speed_to_wrench_gain: Array2x3 = field(
        default_factory=lambda: np.array(
            [[1200.0, 1200.0, 1200.0], [10.0, 10.0, 10.0]],
            dtype=np.float32,
        )
    )
    fixed_axis_near_max_wrench: Array2x3 = field(
        default_factory=lambda: np.array(
            [[30.0, 30.0, 30.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
    )

    # The envelope values are task-specific and injected by tasks/<name>/robot.py.
    rotation_recovery: RotationRecoveryConfig | None = None

    limit: dict[str, Array2x3] = field(
        default_factory=lambda: {
            "max_speed": np.array(
                [[0.08, 0.08, 0.08], [0.30, 0.30, 0.30]], dtype=np.float32
            ),
            # Hard force/torque boundary for the wrench sent to forceMode.
            "max_wrench": np.array(
                [[30.0, 30.0, 30.0], [2.0, 2.0, 4.0]], dtype=np.float32
            ),
        }
    )

    sdk_parameters: dict[str, float] = field(
        default_factory=lambda: {
            "forceMode_damping_factor": 0.000
        }
    )

    def configure_movable_axes(
        self, movable_axes: tuple[bool, bool, bool, bool, bool, bool]
    ) -> None:
        """Apply the shared movable/fixed controller profile in XYZ/RXRYRZ order."""

        movable = np.asarray(movable_axes, dtype=bool)
        if movable.shape != (6,):
            raise ValueError("movable_axes must contain XYZ/RXRYRZ booleans")
        fixed = (~movable).reshape(2, 3)
        self.cascade_gain["pose_to_speed"] = np.where(
            fixed,
            self.fixed_axis_pose_to_speed_gain,
            self.cascade_gain["pose_to_speed"],
        ).astype(np.float32)
        self.cascade_gain["speed_to_wrench"] = np.where(
            fixed,
            self.fixed_axis_speed_to_wrench_gain,
            self.cascade_gain["speed_to_wrench"],
        ).astype(np.float32)
        self.limit["max_wrench"] = np.where(
            fixed,
            self.fixed_axis_max_wrench,
            self.limit["max_wrench"],
        ).astype(np.float32)
        self.soft_pose_zone = np.where(
            fixed, self.fixed_axis_soft_zone, 0.0
        ).astype(np.float32)
        self.soft_pose_to_speed_gain = np.where(
            fixed,
            self.fixed_axis_near_pose_to_speed_gain,
            self.soft_pose_to_speed_gain,
        ).astype(np.float32)
        self.soft_speed_to_wrench_gain = np.where(
            fixed,
            self.fixed_axis_near_speed_to_wrench_gain,
            self.soft_speed_to_wrench_gain,
        ).astype(np.float32)
        self.soft_max_wrench = np.where(
            fixed,
            self.fixed_axis_near_max_wrench,
            self.soft_max_wrench,
        ).astype(np.float32)


class URImpedanceControl:
    def __init__(self, config: URImpedanceControlConfig):
        self.config = config
        self.rtde_r: rtde_receive.REDEReceiveInterface = None
        self.rtde_c: rtde_control.RTDEControlInterface = None
        # The task sets the target after connection/reset. connect() replaces
        # this placeholder with the measured startup pose.
        self.target_pose = np.zeros((2, 3), dtype=np.float32)
        self.rotation_recovery_active = False
        self._validate_rotation_recovery()

    def _validate_rotation_recovery(self) -> None:
        recovery = self.config.rotation_recovery
        if recovery is None:
            return
        half_range = np.asarray(recovery.half_range_rotvec, dtype=np.float32)
        margin = np.asarray(recovery.inward_margin_rotvec, dtype=np.float32)
        activation_delta = np.asarray(
            recovery.activation_delta_rotvec, dtype=np.float32
        )
        max_wrench = np.asarray(recovery.max_wrench, dtype=np.float32)
        if (
            half_range.shape != (3,)
            or margin.shape != (3,)
            or activation_delta.shape != (3,)
            or max_wrench.shape != (3,)
            or np.any(half_range <= 0.0)
            or np.any(margin < 0.0)
            or np.any(margin >= half_range)
            or np.any(activation_delta < 0.0)
            or np.any(max_wrench <= 0.0)
            or recovery.wrench_gain_multiplier < 1.0
        ):
            raise ValueError("invalid rotation recovery configuration")

    def connect(self) -> None:
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.config.ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.config.ip,
                self.config.freq,
                rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
                self.config.port)
        self.rtde_r.waitPeriod(self.rtde_r.initPeriod())
        time.sleep(0.5)
        self.rtde_c.zeroFtSensor()
        time.sleep(0.5)
        # Hold the measured startup pose. Do not move toward the generic
        # configured center before the first explicit actor/reset command.
        self.target_pose = self.get_actual_tcp_pose().copy()
        self.rtde_c.forceModeSetDamping(self.config.\
                sdk_parameters["forceMode_damping_factor"])
        self.rtde_c.forceModeSetGainScaling(1)

    def disconnect(self) -> None:
        self.rtde_c.forceModeStop()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def get_actual_tcp_pose(self) -> Array2x3:
        return self.norm_SO3(np.asarray(self.rtde_r.getActualTCPPose()).reshape(2, 3))

    def get_actual_tcp_speed(self) -> Array2x3:
        return np.asarray(self.rtde_r.getActualTCPSpeed()).reshape(2, 3)

    def get_actual_tcp_force(self) -> Array2x3:
        return np.asarray(self.rtde_r.getActualTCPForce()).reshape(2, 3)

    def set_target_pose(self, target_pose : Array2x3) -> None:
        self.target_pose = np.asarray(target_pose, dtype=np.float32).reshape(2, 3).copy()

    def _relative_rotation(self, rotvec: Array3) -> Array3:
        recovery = self.config.rotation_recovery
        assert recovery is not None
        nominal = R.from_rotvec(np.asarray(recovery.nominal_rotvec, dtype=np.float64))
        actual = R.from_rotvec(np.asarray(rotvec, dtype=np.float64))
        return (nominal.inv() * actual).as_rotvec().astype(np.float32)

    def _rotation_from_relative(self, relative_rotvec: Array3) -> Array3:
        recovery = self.config.rotation_recovery
        assert recovery is not None
        nominal = R.from_rotvec(np.asarray(recovery.nominal_rotvec, dtype=np.float64))
        relative = R.from_rotvec(np.asarray(relative_rotvec, dtype=np.float64))
        return (nominal * relative).as_rotvec().astype(np.float32)

    def constrain_rotation_target(self, candidate_rotvec: Array3) -> Array3:
        """Project an action target onto the normal orientation fence."""

        recovery = self.config.rotation_recovery
        if recovery is None:
            return np.asarray(candidate_rotvec, dtype=np.float32).reshape(3)
        relative = self._relative_rotation(candidate_rotvec)
        half_range = np.asarray(recovery.half_range_rotvec, dtype=np.float32)
        relative = np.clip(relative, -half_range, half_range)
        return self._rotation_from_relative(relative)

    def _recovery_target_rotation(self, actual_rotvec: Array3) -> Array3 | None:
        recovery = self.config.rotation_recovery
        if recovery is None:
            return None
        relative = self._relative_rotation(actual_rotvec)
        half_range = np.asarray(recovery.half_range_rotvec, dtype=np.float32)
        activation_range = half_range + np.asarray(
            recovery.activation_delta_rotvec, dtype=np.float32
        )
        inner_range = half_range - np.asarray(
            recovery.inward_margin_rotvec, dtype=np.float32
        )
        if np.any(np.abs(relative) > activation_range):
            self.rotation_recovery_active = True
        if not self.rotation_recovery_active:
            return None
        if np.all(np.abs(relative) <= half_range):
            self.rotation_recovery_active = False
            return None
        return self._rotation_from_relative(
            np.clip(relative, -inner_range, inner_range)
        )

    def update_rotation_recovery(self, actual_rotvec: Array3) -> bool:
        """Refresh measured-pose recovery state for task-side action gating."""

        self._recovery_target_rotation(actual_rotvec)
        return bool(self.rotation_recovery_active)

    @staticmethod
    def norm_SO3(pose : Array2x3) -> Array2x3:
        return np.asarray([pose[0], R.from_rotvec(pose[1]).as_rotvec()])

    @staticmethod
    def sub_SO3(ori1 : Array3, ori2 : Array3) -> Array3:
        return (R.from_rotvec(ori1) * R.from_rotvec(ori2).inv()).as_rotvec()

    def _optional_rtde_vector(
        self, method_name: str, size: int
    ) -> NDArray[np.float32]:
        """Read diagnostic-only RTDE data without affecting control."""

        method = getattr(self.rtde_r, method_name, None)
        if not callable(method):
            return np.full(size, np.nan, dtype=np.float32)
        try:
            value = np.asarray(method(), dtype=np.float32).reshape(-1)
        except Exception:
            return np.full(size, np.nan, dtype=np.float32)
        if value.shape != (size,):
            return np.full(size, np.nan, dtype=np.float32)
        return value.copy()

    def pid_step(self, *, diagnostics: bool = False) -> dict[str, object] | None:
        pose_to_speed = self.config.cascade_gain["pose_to_speed"]
        speed_to_wrench = self.config.cascade_gain["speed_to_wrench"]
        force_mode_limits = self.config.limit["max_speed"].reshape(-1).copy()
        max_wrench = np.asarray(self.config.limit["max_wrench"], dtype=np.float32).copy()

        # Read the measured Cartesian pose and twist from UR RTDE.
        actual_pose         = self.get_actual_tcp_pose()
        actual_speed        = self.get_actual_tcp_speed()

        was_recovering = self.rotation_recovery_active
        effective_target_pose = self.target_pose.copy()
        recovery_rotation = self._recovery_target_rotation(actual_pose[1])
        if was_recovering and not self.rotation_recovery_active:
            # Resume from the measured pose instead of jumping toward a stale
            # translation/rotation target accumulated before recovery.
            self.target_pose = actual_pose.copy()
            effective_target_pose = actual_pose.copy()
        if recovery_rotation is not None:
            # Recovery-only mode: no translation target is active.
            effective_target_pose[0] = actual_pose[0]
            effective_target_pose[1] = recovery_rotation
            recovery = self.config.rotation_recovery
            assert recovery is not None
            max_wrench[1] = np.asarray(recovery.max_wrench, dtype=np.float32)

        # Outer pose loop: target pose -> bounded desired Cartesian twist.
        raw_err_pose = np.asarray(
            [
                effective_target_pose[0] - actual_pose[0],
                self.sub_SO3(effective_target_pose[1], actual_pose[1]),
            ]
        )
        deadzone = np.asarray(
            self.config.pose_error_deadzone, dtype=np.float32
        ).reshape(2, 3)
        err_pose = np.sign(raw_err_pose) * np.maximum(
            np.abs(raw_err_pose) - deadzone,
            0.0,
        )
        active_pose_to_speed = np.asarray(pose_to_speed, dtype=np.float32)
        active_speed_to_wrench = np.asarray(speed_to_wrench, dtype=np.float32)
        if recovery_rotation is None:
            soft_zone = np.asarray(
                self.config.soft_pose_zone, dtype=np.float32
            ).reshape(2, 3)
            soft_mask = (soft_zone > 0.0) & (
                np.abs(raw_err_pose) < soft_zone
            )
            active_pose_to_speed = np.where(
                soft_mask,
                np.asarray(self.config.soft_pose_to_speed_gain),
                active_pose_to_speed,
            )
            active_speed_to_wrench = np.where(
                soft_mask,
                np.asarray(self.config.soft_speed_to_wrench_gain),
                active_speed_to_wrench,
            )
            max_wrench = np.where(
                soft_mask,
                np.asarray(self.config.soft_max_wrench),
                max_wrench,
            )
        target_speed = np.clip(
            active_pose_to_speed * err_pose,
            -self.config.limit["max_speed"],
            self.config.limit["max_speed"],
        )

        # Inner velocity loop: twist error -> Cartesian wrench.
        err_speed = target_speed - actual_speed
        controller_wrench = active_speed_to_wrench * err_speed
        if recovery_rotation is not None:
            recovery = self.config.rotation_recovery
            assert recovery is not None
            controller_wrench[1] *= float(recovery.wrench_gain_multiplier)
            controller_wrench[0].fill(0.0)

        # Bound each force/torque component, then send it directly to
        # forceMode. No manual force-to-torque compensation is applied.
        target_tcp_wrench = np.clip(
            controller_wrench,
            -max_wrench,
            max_wrench,
        )

        # Every Cartesian axis remains force-compliant. Translation-only tasks
        # still provide an upright rotational target, but it is tracked through
        # bounded torque rather than UR's non-compliant position lock.
        selection_vector = np.ones(6, dtype=np.int32)
        self.rtde_c.forceMode(\
                np.zeros(6, dtype=np.float32), \
                selection_vector, \
                target_tcp_wrench.reshape(-1), \
                2,
                force_mode_limits)
        if not diagnostics:
            return None
        return {
            "monotonic_ns": time.monotonic_ns(),
            "actual_tcp_pose": actual_pose.reshape(6).astype(np.float32).copy(),
            "actual_tcp_speed": actual_speed.reshape(6).astype(np.float32).copy(),
            "actual_tcp_force": self.get_actual_tcp_force()
            .reshape(6)
            .astype(np.float32)
            .copy(),
            "actual_q": self._optional_rtde_vector("getActualQ", 6),
            "actual_qd": self._optional_rtde_vector("getActualQd", 6),
            "target_qdd": self._optional_rtde_vector("getTargetQdd", 6),
            "tool_accelerometer": self._optional_rtde_vector(
                "getActualToolAccelerometer", 3
            ),
            "effective_target_pose": effective_target_pose.reshape(6)
            .astype(np.float32)
            .copy(),
            "raw_pose_error": raw_err_pose.reshape(6).astype(np.float32).copy(),
            "pose_error": err_pose.reshape(6).astype(np.float32).copy(),
            "target_tcp_speed": target_speed.reshape(6).astype(np.float32).copy(),
            "speed_error": err_speed.reshape(6).astype(np.float32).copy(),
            "pid_wrench": controller_wrench.reshape(6).astype(np.float32).copy(),
            "commanded_tcp_wrench": target_tcp_wrench.reshape(6)
            .astype(np.float32)
            .copy(),
            "rotation_recovery_active": bool(self.rotation_recovery_active),
        }
