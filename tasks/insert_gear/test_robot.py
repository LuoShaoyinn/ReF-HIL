from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from .robot import Robot, RobotConfig


class _ImmediateImpedanceControl:
    def __init__(self, pose: np.ndarray) -> None:
        self.pose = pose.copy()
        self.targets: list[np.ndarray] = []

    def constrain_rotation_target(self, rotation: np.ndarray) -> np.ndarray:
        return np.asarray(rotation, dtype=np.float32)

    def set_target_pose(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float32).copy()
        self.targets.append(self.pose.copy())

    def get_actual_tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def update_rotation_recovery(self, rotation):
        return False


class InsertGearRobotTest(unittest.TestCase):
    def test_complete_reset_through_driver_preserves_open_gripper_and_reset_fence(self):
        from .modeling import Modeling, ModelingConfig, rotation_at_yaw

        robot = object.__new__(Robot)
        robot.config = RobotConfig(fake_mode=True)
        robot._startup_complete = True
        robot.gripper = object()
        task = robot.config.task
        robot.impedance_control = _ImmediateImpedanceControl(np.asarray([
            [-0.120, -0.470, 0.060], rotation_at_yaw(task, 0.5)], dtype=np.float32))
        modeling = object.__new__(Modeling)
        modeling.config = ModelingConfig(device="cpu")
        with patch("tasks.insert_gear.robot.send_continuous_gripper_action") as grip, patch("tasks.insert_gear.modeling.time.sleep"):
            modeling.reset(robot.apply_action, lambda: {"tcp_pose": robot.impedance_control.get_actual_tcp_pose()})
        targets = robot.impedance_control.targets
        np.testing.assert_allclose(targets[0][0], [-0.120, -0.470, task.reset_release_z])
        np.testing.assert_allclose(targets[1][0], [-0.120, -0.470, task.reset_pos[2]])
        self.assertTrue(
            any(
                np.allclose(target[0], task.startup_pickup_pos)
                for target in targets
            )
        )
        values = [call.args[1] for call in grip.call_args_list]
        first_open = values.index(-1.0)
        last_open = len(values) - 1 - values[::-1].index(-1.0)
        self.assertTrue(all(v == -1.0 for v in values[first_open:last_open + 1]))
        self.assertTrue(all(v == 1.0 for v in values[last_open + 1:]))

    def test_reset_opens_and_reaches_release_height_without_reclosing(self):
        robot = object.__new__(Robot)
        robot.config = RobotConfig(fake_mode=True)
        robot._startup_complete = True
        robot.gripper = object()
        task = robot.config.task
        robot.impedance_control = _ImmediateImpedanceControl(np.asarray([task.reset_pos, task.zero_point_rot]))
        with patch("tasks.insert_gear.robot.send_continuous_gripper_action") as grip, patch.object(robot, "_close_gripper") as close:
            for command in (-1.0, 1.0):
                robot.apply_action({"insert_gear_reset": True, "gripper_only": True, "gripper": command})
                grip.assert_called_with(robot.gripper, command, speed=task.gripper_speed, force=task.gripper_force)
            release_pos = [*task.reset_pos[:2], task.reset_release_z]
            robot.apply_action({"insert_gear_reset": True, "reset_target_pose": [release_pos, task.zero_point_rot], "gripper": -1.0})
            np.testing.assert_allclose(robot.impedance_control.targets[-1][0], release_pos)
            grip.assert_called_with(robot.gripper, -1.0, speed=task.gripper_speed, force=task.gripper_force)
            close.assert_not_called()

    def test_ordinary_actions_clip_to_workspace_and_keep_gripper_closed(self):
        robot = object.__new__(Robot)
        robot.config = RobotConfig(fake_mode=True)
        robot._startup_complete = True
        task = robot.config.task
        robot.impedance_control = _ImmediateImpedanceControl(np.asarray([task.reset_pos, task.zero_point_rot]))
        with patch.object(robot, "_close_gripper") as close:
            robot.apply_action({"reset_target_pose": [[0.100, -0.300, 0.121], task.zero_point_rot], "gripper": -1.0})
            self.assertAlmostEqual(float(robot.impedance_control.targets[-1][0, 0]), task.safety_pos_max[0])
            self.assertAlmostEqual(float(robot.impedance_control.targets[-1][0, 1]), task.safety_pos_max[1])
            close.assert_called_once()

    def test_rz_control_rates_match_rotate_knob(self) -> None:
        config = RobotConfig(fake_mode=True)
        self.assertEqual(
            config.impendence_control.limit["max_speed"][1, 2],
            config.task.yaw_speed_limit_rad_s,
        )

    def test_interactive_pickup_approaches_closes_then_lifts(self) -> None:
        robot = object.__new__(Robot)
        robot.config = RobotConfig(fake_mode=True)
        task = robot.config.task
        initial = np.asarray(
            [task.reset_pos, task.zero_point_rot], dtype=np.float32
        )
        robot.impedance_control = _ImmediateImpedanceControl(initial)
        robot._startup_complete = False

        with (
            patch("builtins.input", side_effect=["", "", ""]),
            patch.object(robot, "_close_gripper") as close_gripper,
        ):
            robot._interactive_pickup()

        targets = robot.impedance_control.targets
        self.assertEqual(len(targets), 3)
        np.testing.assert_allclose(
            targets[0][0],
            [task.startup_pickup_pos[0], task.startup_pickup_pos[1], task.reset_pos[2]],
        )
        np.testing.assert_allclose(targets[1][0], task.startup_pickup_pos)
        np.testing.assert_allclose(
            targets[2][0],
            [task.startup_pickup_pos[0], task.startup_pickup_pos[1], task.reset_pos[2]],
        )
        close_gripper.assert_called_once_with()
        self.assertTrue(robot._startup_complete)


if __name__ == "__main__":
    unittest.main()
