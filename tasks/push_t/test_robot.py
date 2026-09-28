from __future__ import annotations

import sys
from types import ModuleType
import unittest

import numpy as np

sys.modules.setdefault("rtde_control", ModuleType("rtde_control"))
sys.modules.setdefault("rtde_receive", ModuleType("rtde_receive"))
sys.modules.setdefault("pyrealsense2", ModuleType("pyrealsense2"))

from .robot import Robot, RobotConfig


class _ImpedanceControl:
    def __init__(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float32).reshape(2, 3)
        self.target_pose = self.pose.copy()

    def get_actual_tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def update_rotation_recovery(self, _rotation: np.ndarray) -> bool:
        return False

    def constrain_rotation_target(self, rotation: np.ndarray) -> np.ndarray:
        return np.asarray(rotation, dtype=np.float32)

    def set_target_pose(self, pose: np.ndarray) -> None:
        self.target_pose = np.asarray(pose, dtype=np.float32).copy()


class _Gripper:
    def __init__(self) -> None:
        self.commands: list[tuple[int, int, int]] = []

    def get_open_position(self) -> int:
        return 0

    def get_closed_position(self) -> int:
        return 255

    def move(self, position: int, speed: int, force: int) -> None:
        self.commands.append((position, speed, force))


class PushTRobotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = Robot.__new__(Robot)
        self.robot.config = RobotConfig()
        task = self.robot.config.task
        pose = np.asarray(
            [[-0.15, -0.545, 0.04], task.zero_point_rot],
            dtype=np.float32,
        )
        self.robot.impedance_control = _ImpedanceControl(pose)
        self.robot.gripper = _Gripper()

    def test_fixed_rotation_uses_shared_holding_profile(self) -> None:
        control = self.robot.config.impendence_control
        np.testing.assert_allclose(
            control.cascade_gain["pose_to_speed"],
            [[20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
        )
        np.testing.assert_allclose(
            control.soft_pose_zone,
            [[0.0, 0.0, 0.0], [np.deg2rad(5.0)] * 3],
        )
        np.testing.assert_allclose(
            control.limit["max_wrench"],
            [[30.0, 30.0, 30.0], [4.0, 4.0, 4.0]],
        )

    def test_push_t_uses_high_resolution_global_camera_only(self) -> None:
        cameras = self.robot.config.realsense
        self.assertEqual((cameras["global"].width, cameras["global"].height), (1280, 720))
        self.assertEqual((cameras["wrist_0"].width, cameras["wrist_0"].height), (640, 480))

    def test_action_controls_xyz_and_holds_fixed_rotation(self) -> None:
        task = self.robot.config.task
        self.robot.apply_action({
            "delta_pos": np.asarray([1.0, -1.0, 0.5]),
            "delta_rot": np.asarray([1.0, -1.0, 0.5]),
            "gripper": -1.0,
        })

        target = self.robot.impedance_control.target_pose
        np.testing.assert_allclose(
            target[0],
            [-0.10, -0.595, 0.060],
            atol=1e-6,
        )
        np.testing.assert_allclose(target[1], task.zero_point_rot, atol=1e-6)
        self.assertEqual(self.robot.gripper.commands[-1][0], 255)

    def test_xyz_targets_are_clipped_to_task_bounds(self) -> None:
        task = self.robot.config.task
        self.robot.impedance_control.pose[0] = np.asarray(
            [-0.099, -0.299, 0.059]
        )

        self.robot.apply_action({
            "delta_pos": np.ones(3),
            "delta_rot": np.ones(3),
        })

        target = self.robot.impedance_control.target_pose
        self.assertAlmostEqual(target[0, 0], task.safety_pos_max[0])
        self.assertAlmostEqual(target[0, 1], task.safety_pos_max[1])
        self.assertAlmostEqual(target[0, 2], task.safety_pos_max[2])
        np.testing.assert_allclose(target[1], task.zero_point_rot, atol=1e-6)

    def test_reset_target_uses_fixed_orientation(self) -> None:
        task = self.robot.config.task
        reset_target = np.asarray(
            [
                [-0.20, -0.60, task.reset_lift_z],
                task.zero_point_rot,
            ],
            dtype=np.float32,
        )

        self.robot.apply_action({"reset_target_pose": reset_target})

        np.testing.assert_allclose(
            self.robot.impedance_control.target_pose,
            reset_target,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
