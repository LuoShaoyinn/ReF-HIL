from __future__ import annotations

import sys
from types import ModuleType
import unittest

import numpy as np


# Hardware SDKs are intentionally absent on learner-only machines. The
# controller math below does not require them, so provide import-only stubs.
sys.modules.setdefault("rtde_receive", ModuleType("rtde_receive"))
sys.modules.setdefault("rtde_control", ModuleType("rtde_control"))

from actor.robot.ur5e.ur_impedance_control import (  # noqa: E402
    RotationRecoveryConfig,
    URImpedanceControl,
    URImpedanceControlConfig,
)


class _FakeRTDEReceive:
    def __init__(
        self,
        pose: np.ndarray,
        speed: np.ndarray,
        force: np.ndarray | None = None,
    ) -> None:
        self.pose = pose
        self.speed = speed
        self.force = (
            np.zeros((2, 3), dtype=np.float32) if force is None else force
        )

    def getActualTCPPose(self) -> np.ndarray:
        return self.pose.reshape(-1)

    def getActualTCPSpeed(self) -> np.ndarray:
        return self.speed.reshape(-1)

    def getActualTCPForce(self) -> np.ndarray:
        return self.force.reshape(-1)


class _FakeRTDEControl:
    def __init__(self) -> None:
        self.wrench: np.ndarray | None = None
        self.speed_limit: np.ndarray | None = None

    def forceMode(
        self,
        _frame: np.ndarray,
        _selection: np.ndarray,
        wrench: np.ndarray,
        _mode: int,
        speed_limit: np.ndarray,
    ) -> None:
        self.wrench = np.asarray(wrench)
        self.speed_limit = np.asarray(speed_limit)


class CascadedImpedanceControlTest(unittest.TestCase):
    @staticmethod
    def _recovery_config() -> URImpedanceControlConfig:
        return URImpedanceControlConfig(
            rotation_recovery=RotationRecoveryConfig(
                nominal_rotvec=(0.0, 0.0, 0.0),
                half_range_rotvec=(0.20, 0.20, 0.20),
                inward_margin_rotvec=(0.05, 0.05, 0.05),
                activation_delta_rotvec=(0.05, 0.05, 0.05),
                max_wrench=(4.0, 4.0, 4.0),
                wrench_gain_multiplier=2.0,
            )
        )

    def test_pose_to_speed_to_wrench_respects_force_level_clip(self) -> None:
        controller = URImpedanceControl(URImpedanceControlConfig())
        actual_pose = np.zeros((2, 3), dtype=np.float32)
        actual_speed = np.zeros((2, 3), dtype=np.float32)
        controller.target_pose = np.asarray(
            [[1.0, -1.0, 1.0], [0.5, -0.5, 0.5]], dtype=np.float32
        )
        controller.rtde_r = _FakeRTDEReceive(actual_pose, actual_speed)
        controller.rtde_c = _FakeRTDEControl()

        controller.pid_step()

        wrench = controller.rtde_c.wrench.reshape(2, 3)
        self.assertTrue(
            np.all(np.abs(wrench) <= controller.config.limit["max_wrench"])
        )
        np.testing.assert_allclose(
            controller.rtde_c.speed_limit,
            controller.config.limit["max_speed"].reshape(-1),
        )
        self.assertGreater(wrench[0, 0], 0.0)
        self.assertLess(wrench[0, 1], 0.0)
        self.assertGreater(wrench[1, 0], 0.0)

    def test_translation_force_does_not_inject_rotation_torque(self) -> None:
        controller = URImpedanceControl(URImpedanceControlConfig())
        actual_pose = np.zeros((2, 3), dtype=np.float32)
        actual_speed = np.zeros((2, 3), dtype=np.float32)
        controller.target_pose = actual_pose.copy()
        controller.target_pose[0, 0] = 1.0
        controller.rtde_r = _FakeRTDEReceive(actual_pose, actual_speed)
        controller.rtde_c = _FakeRTDEControl()
        controller.pid_step()
        wrench = controller.rtde_c.wrench.reshape(2, 3)
        np.testing.assert_allclose(wrench[0], [30.0, 0.0, 0.0])
        np.testing.assert_allclose(wrench[1], np.zeros(3))

    def test_pose_deadzone_is_continuous_and_axis_specific(self) -> None:
        config = URImpedanceControlConfig()
        config.pose_error_deadzone[1, 0] = np.deg2rad(1.0)
        controller = URImpedanceControl(config)
        actual_pose = np.zeros((2, 3), dtype=np.float32)
        actual_speed = np.zeros((2, 3), dtype=np.float32)
        controller.rtde_r = _FakeRTDEReceive(actual_pose, actual_speed)
        controller.rtde_c = _FakeRTDEControl()

        controller.target_pose = actual_pose.copy()
        controller.target_pose[1, 0] = np.deg2rad(0.5)
        controller.pid_step()
        self.assertAlmostEqual(
            float(controller.rtde_c.wrench.reshape(2, 3)[1, 0]), 0.0
        )

        controller.target_pose[1, 0] = np.deg2rad(1.5)
        controller.pid_step()
        expected = (
            config.cascade_gain["pose_to_speed"][1, 0]
            * np.deg2rad(0.5)
            * config.cascade_gain["speed_to_wrench"][1, 0]
        )
        self.assertAlmostEqual(
            float(controller.rtde_c.wrench.reshape(2, 3)[1, 0]),
            float(expected),
            places=5,
        )

    def test_soft_pose_zone_switches_to_original_pid_near_target(self) -> None:
        config = URImpedanceControlConfig()
        config.cascade_gain["pose_to_speed"][1, 0] = 30.0
        config.cascade_gain["speed_to_wrench"][1, 0] = 20.0
        config.limit["max_wrench"][1, 0] = 4.0
        config.soft_pose_zone[1, 0] = np.deg2rad(5.0)
        controller = URImpedanceControl(config)
        actual_pose = np.zeros((2, 3), dtype=np.float32)
        actual_speed = np.zeros((2, 3), dtype=np.float32)
        controller.rtde_r = _FakeRTDEReceive(actual_pose, actual_speed)
        controller.rtde_c = _FakeRTDEControl()

        controller.target_pose = actual_pose.copy()
        controller.target_pose[1, 0] = np.deg2rad(4.0)
        controller.pid_step()
        self.assertAlmostEqual(
            float(controller.rtde_c.wrench.reshape(2, 3)[1, 0]),
            2.0,
            places=5,
        )

        controller.target_pose[1, 0] = np.deg2rad(6.0)
        controller.pid_step()
        self.assertAlmostEqual(
            float(controller.rtde_c.wrench.reshape(2, 3)[1, 0]),
            4.0,
            places=5,
        )

    def test_movable_axis_mask_applies_fixed_profile_only_to_complement(self) -> None:
        config = URImpedanceControlConfig()
        config.configure_movable_axes(
            (True, True, False, False, False, True)
        )
        np.testing.assert_allclose(
            config.cascade_gain["pose_to_speed"],
            [[20.0, 20.0, 30.0], [30.0, 30.0, 20.0]],
        )
        np.testing.assert_allclose(
            config.cascade_gain["speed_to_wrench"],
            [[1200.0, 1200.0, 1200.0], [20.0, 20.0, 40.0]],
        )
        np.testing.assert_allclose(
            config.soft_pose_zone,
            [[0.0, 0.0, 0.005], [np.deg2rad(5.0), np.deg2rad(5.0), 0.0]],
        )
        np.testing.assert_allclose(
            config.soft_speed_to_wrench_gain,
            [[1200.0, 1200.0, 1200.0], [10.0, 10.0, 40.0]],
        )
        np.testing.assert_allclose(
            config.limit["max_wrench"],
            [[30.0, 30.0, 30.0], [4.0, 4.0, 2.0]],
        )

    def test_fixed_yaw_uses_soft_near_gain_not_movable_yaw_gain(self) -> None:
        config = URImpedanceControlConfig()
        config.configure_movable_axes(
            (True, True, True, False, False, False)
        )
        np.testing.assert_allclose(
            config.soft_speed_to_wrench_gain[1],
            [10.0, 10.0, 10.0],
        )
        np.testing.assert_allclose(
            config.soft_max_wrench[1],
            [2.0, 2.0, 2.0],
        )

    def test_diagnostics_report_sensor_and_wrench_paths(self) -> None:
        controller = URImpedanceControl(URImpedanceControlConfig())
        actual_pose = np.zeros((2, 3), dtype=np.float32)
        actual_speed = np.zeros((2, 3), dtype=np.float32)
        controller.target_pose = actual_pose.copy()
        controller.rtde_r = _FakeRTDEReceive(actual_pose, actual_speed)
        controller.rtde_c = _FakeRTDEControl()

        diagnostic = controller.pid_step(diagnostics=True)

        assert diagnostic is not None
        self.assertEqual(np.asarray(diagnostic["actual_tcp_force"]).shape, (6,))
        self.assertEqual(np.asarray(diagnostic["pid_wrench"]).shape, (6,))
        self.assertEqual(
            np.asarray(diagnostic["commanded_tcp_wrench"]).shape, (6,)
        )
        self.assertTrue(np.isnan(np.asarray(diagnostic["target_qdd"])).all())

    def test_crossing_action_is_projected_without_latching_recovery(self) -> None:
        controller = URImpedanceControl(self._recovery_config())

        constrained = controller.constrain_rotation_target(
            np.asarray([0.25, 0.0, 0.0], dtype=np.float32)
        )

        self.assertFalse(controller.rotation_recovery_active)
        np.testing.assert_allclose(constrained, [0.20, 0.0, 0.0], atol=1e-6)

    def test_measured_pose_inside_activation_delta_uses_normal_control(self) -> None:
        controller = URImpedanceControl(self._recovery_config())

        active = controller.update_rotation_recovery(
            np.asarray([0.24, 0.0, 0.0], dtype=np.float32)
        )

        self.assertFalse(active)

    def test_measured_violation_uses_strong_bounded_inward_torque(self) -> None:
        controller = URImpedanceControl(self._recovery_config())
        actual_pose = np.asarray(
            [[0.0, 0.0, 0.0], [0.26, 0.0, 0.0]], dtype=np.float32
        )
        controller.target_pose = np.asarray(
            [[1.0, -1.0, 1.0], [0.26, 0.0, 0.0]], dtype=np.float32
        )
        controller.rtde_r = _FakeRTDEReceive(
            actual_pose,
            np.asarray([[0.05, -0.05, 0.05], [0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[20.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        )
        controller.rtde_c = _FakeRTDEControl()

        controller.pid_step()

        wrench = controller.rtde_c.wrench.reshape(2, 3)
        self.assertTrue(controller.rotation_recovery_active)
        np.testing.assert_array_equal(wrench[0], np.zeros(3))
        self.assertAlmostEqual(float(wrench[1, 0]), -4.0, places=6)
        np.testing.assert_allclose(wrench[1, 1:], [0.0, 0.0], atol=1e-6)

    def test_recovery_releases_only_after_entering_inset_range(self) -> None:
        controller = URImpedanceControl(self._recovery_config())
        controller.rotation_recovery_active = True
        actual_pose = np.asarray(
            [[0.0, 0.0, 0.0], [0.19, 0.0, 0.0]], dtype=np.float32
        )
        controller.target_pose = np.asarray(
            [[1.0, -1.0, 1.0], [-0.10, 0.0, 0.0]], dtype=np.float32
        )
        controller.rtde_r = _FakeRTDEReceive(
            actual_pose, np.zeros((2, 3), dtype=np.float32)
        )
        controller.rtde_c = _FakeRTDEControl()

        controller.pid_step()

        self.assertFalse(controller.rotation_recovery_active)
        np.testing.assert_allclose(controller.target_pose, actual_pose, atol=1e-6)
        np.testing.assert_allclose(
            controller.rtde_c.wrench.reshape(2, 3)[1],
            np.zeros(3),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
