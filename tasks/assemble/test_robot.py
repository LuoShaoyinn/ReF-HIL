from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("rtde_control", ModuleType("rtde_control"))
sys.modules.setdefault("rtde_receive", ModuleType("rtde_receive"))
sys.modules.setdefault("pyrealsense2", ModuleType("pyrealsense2"))

from .robot import Robot, RobotConfig, normalized_gripper_position
from actor.robot.ur5e import UR5eBase


class _ImpedanceControl:
    def __init__(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float32).reshape(2, 3)
        self.target_pose: np.ndarray | None = None

    def get_actual_tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def set_target_pose(self, pose: np.ndarray) -> None:
        self.target_pose = np.asarray(pose, dtype=np.float32).copy()


class AssembleRobotTest(unittest.TestCase):
    def test_policy_gripper_uses_restricted_continuous_travel(self) -> None:
        self.assertEqual(normalized_gripper_position(-1.0, 0, 255, (0.75, 1.0)), 0)
        self.assertEqual(normalized_gripper_position(1.0, 0, 255, (0.75, 1.0)), 64)
        self.assertEqual(normalized_gripper_position(0.0, 0, 255, (0.75, 1.0)), 32)

    def test_assemble_keeps_rotation_recovery_disabled(self) -> None:
        config = RobotConfig()
        self.assertIsNone(config.impendence_control.rotation_recovery)

    def test_translation_only_action_targets_canonical_upright_rotation(self) -> None:
        robot = Robot.__new__(Robot)
        robot.config = RobotConfig()
        robot.impedance_control = _ImpedanceControl(
            np.asarray([[0.0, 0.0, 0.05], [0.1, -0.2, 0.3]])
        )
        robot.apply_action(
            {
                "delta_pos": np.zeros(3, dtype=np.float32),
                "delta_rot": np.ones(3, dtype=np.float32),
            }
        )
        assert robot.impedance_control.target_pose is not None
        np.testing.assert_allclose(
            robot.impedance_control.target_pose[1],
            robot.config.task.zero_point_rot,
        )

    def test_connect_immediately_targets_canonical_upright_rotation(self) -> None:
        robot = Robot.__new__(Robot)
        robot.config = RobotConfig()
        robot.impedance_control = _ImpedanceControl(
            np.asarray([[0.0, 0.0, 0.05], [0.1, -0.2, 0.3]])
        )
        with patch.object(UR5eBase, "connect", autospec=True):
            robot.connect()
        assert robot.impedance_control.target_pose is not None
        np.testing.assert_allclose(
            robot.impedance_control.target_pose[1],
            robot.config.task.zero_point_rot,
        )


if __name__ == "__main__":
    unittest.main()
