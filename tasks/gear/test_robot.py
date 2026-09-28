from __future__ import annotations

import sys
from types import ModuleType
import unittest

import numpy as np

sys.modules.setdefault("rtde_control", ModuleType("rtde_control"))
sys.modules.setdefault("rtde_receive", ModuleType("rtde_receive"))
sys.modules.setdefault("pyrealsense2", ModuleType("pyrealsense2"))

from .modeling import rotation_at_yaw, yaw_relative_to_nominal
from .robot import Robot, RobotConfig


class _Gripper:
    def __init__(self) -> None:
        self.commands: list[tuple[int, int, int]] = []

    def get_open_position(self) -> int:
        return 0

    def get_closed_position(self) -> int:
        return 255

    def move(self, position: int, speed: int, force: int) -> None:
        self.commands.append((position, speed, force))


class _ImpedanceControl:
    def __init__(self) -> None:
        self.pose = np.asarray(
            [[-0.20, -0.50, 0.10], [2.221, 2.221, 0.0]], dtype=np.float32
        )
        self.target_pose = self.pose.copy()

    def get_actual_tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def update_rotation_recovery(self, _rotation: np.ndarray) -> bool:
        return False

    def constrain_rotation_target(self, rotation: np.ndarray) -> np.ndarray:
        return np.asarray(rotation, dtype=np.float32)

    def set_target_pose(self, pose: np.ndarray) -> None:
        self.target_pose = np.asarray(pose, dtype=np.float32).copy()


class GearRobotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = Robot.__new__(Robot)
        self.robot.config = RobotConfig()
        self.robot.impedance_control = _ImpedanceControl()
        self.robot.gripper = _Gripper()

    def test_policy_controls_xyz_yaw_and_gripper(self) -> None:
        self.robot.apply_action(
            {
                "delta_pos": np.asarray([1.0, -1.0, 1.0]),
                "delta_rot": np.asarray([1.0, 1.0, -1.0]),
                "gripper": 0.0,
            }
        )
        target = self.robot.impedance_control.target_pose
        np.testing.assert_allclose(target[0], [-0.15, -0.55, 0.15])
        self.assertAlmostEqual(
            yaw_relative_to_nominal(self.robot.config.task, target[1]),
            -self.robot.config.task.angular_speed,
            places=5,
        )
        self.assertEqual(self.robot.gripper.commands[-1][0], 128)

    def test_repeated_yaw_commands_accumulate_the_commanded_target(self) -> None:
        action = {
            "delta_pos": np.zeros(3, dtype=np.float32),
            "delta_rot": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        }
        self.robot.apply_action(action)
        first_yaw = yaw_relative_to_nominal(
            self.robot.config.task,
            self.robot.impedance_control.target_pose[1],
        )
        # The fake measured pose deliberately remains unchanged. The command
        # The yaw target accumulates independently of measured-pose lag.
        self.robot.apply_action(action)
        second_yaw = yaw_relative_to_nominal(
            self.robot.config.task,
            self.robot.impedance_control.target_pose[1],
        )
        self.assertAlmostEqual(first_yaw, -self.robot.config.task.angular_speed)
        self.assertAlmostEqual(
            second_yaw,
            -2.0 * self.robot.config.task.angular_speed,
            places=6,
        )

    def test_yaw_target_is_clipped_to_minus_180_and_zero_degrees(self) -> None:
        task = self.robot.config.task
        for starting_yaw, command, expected in (
            (-0.02, 1.0, 0.0),
            (-np.pi + 0.02, -1.0, -np.pi),
        ):
            self.robot.impedance_control.target_pose[1] = rotation_at_yaw(
                task, starting_yaw
            )
            self.robot.apply_action(
                {
                    "delta_pos": np.zeros(3, dtype=np.float32),
                    "delta_rot": np.asarray(
                        [0.0, 0.0, command], dtype=np.float32
                    ),
                }
            )
            target_yaw = yaw_relative_to_nominal(
                task, self.robot.impedance_control.target_pose[1]
            )
            self.assertAlmostEqual(target_yaw, expected, places=5)

    def test_fixed_roll_pitch_are_strong_far_and_soft_near(self) -> None:
        config = self.robot.config
        np.testing.assert_allclose(
            config.impendence_control.cascade_gain["pose_to_speed"][1],
            [30.0, 30.0, 20.0],
        )
        np.testing.assert_allclose(
            config.impendence_control.cascade_gain["speed_to_wrench"][1],
            [20.0, 20.0, 40.0],
        )
        np.testing.assert_allclose(
            config.impendence_control.limit["max_wrench"][1],
            [4.0, 4.0, 2.0],
        )
        np.testing.assert_allclose(
            config.impendence_control.pose_error_deadzone[1],
            np.zeros(3),
        )
        np.testing.assert_allclose(
            config.impendence_control.soft_pose_zone[1],
            [np.deg2rad(5.0), np.deg2rad(5.0), 0.0],
        )
        np.testing.assert_allclose(
            config.impendence_control.soft_pose_to_speed_gain[1, :2],
            [20.0, 20.0],
        )
        np.testing.assert_allclose(
            config.impendence_control.soft_speed_to_wrench_gain[1, :2],
            [10.0, 10.0],
        )
        np.testing.assert_allclose(
            config.impendence_control.soft_max_wrench[1, :2],
            [2.0, 2.0],
        )
        recovery = config.impendence_control.rotation_recovery
        assert recovery is not None
        np.testing.assert_allclose(recovery.max_wrench[:2], [4.0, 4.0])
        self.assertEqual(recovery.max_wrench[2], 4.0)

    def test_movable_yaw_retains_normal_compliant_pid(self) -> None:
        config = self.robot.config
        self.assertEqual(
            config.impendence_control.cascade_gain["pose_to_speed"][1, 2],
            20.0,
        )
        self.assertEqual(
            config.impendence_control.cascade_gain["speed_to_wrench"][1, 2],
            40.0,
        )
        self.assertAlmostEqual(
            config.impendence_control.limit["max_speed"][1, 2],
            0.30,
            places=6,
        )
        self.assertEqual(
            config.impendence_control.limit["max_wrench"][1, 2],
            2.0,
        )

    def test_position_target_is_projected_into_safety_box(self) -> None:
        self.robot.impedance_control.pose[0] = [-0.295, -0.615, 0.145]
        self.robot.apply_action(
            {
                "delta_pos": np.ones(3, dtype=np.float32),
                "delta_rot": np.zeros(3, dtype=np.float32),
            }
        )
        np.testing.assert_allclose(
            self.robot.impedance_control.target_pose[0],
            [-0.245, -0.565, 0.15],
        )

    def test_gripper_only_command_does_not_change_pose_target(self) -> None:
        original_target = self.robot.impedance_control.target_pose.copy()
        self.robot.apply_action({"gripper_only": True, "gripper": -1.0})
        np.testing.assert_allclose(
            self.robot.impedance_control.target_pose, original_target
        )
        self.assertEqual(self.robot.gripper.commands[-1][0], 0)


if __name__ == "__main__":
    unittest.main()
