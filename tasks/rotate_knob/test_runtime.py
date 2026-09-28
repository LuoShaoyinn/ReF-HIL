"""CPU-only contracts with fake hardware; never connects to a robot."""

import math
import sys
from types import ModuleType
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

for name in ("rtde_control", "rtde_receive", "pyrealsense2"):
    sys.modules.setdefault(name, ModuleType(name))

from .modeling import Modeling, ModelingConfig, rotation_at_yaw, yaw_relative_to_nominal
from .robot import Robot, RobotConfig


class FakeGripper:
    def __init__(self):
        self.commands = []

    def get_open_position(self):
        return 0

    def get_closed_position(self):
        return 255

    def move(self, position, speed, force):
        self.commands.append((position, speed, force))


class FakeControl:
    def __init__(self, task):
        self.pose = np.array([task.reset_pos, task.zero_point_rot], dtype=np.float32)
        self.target_pose = self.pose.copy()
        self.recovering = False

    def get_actual_tcp_pose(self):
        return self.pose.copy()

    def update_rotation_recovery(self, rotation):
        return self.recovering

    def constrain_rotation_target(self, rotation):
        return np.asarray(rotation)

    def set_target_pose(self, pose):
        self.target_pose = np.asarray(pose).copy()


class KnobRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(Modeling)
        self.model.config = ModelingConfig(device="cpu")
        self.model.device = torch.device("cpu")
        self.robot = object.__new__(Robot)
        self.robot.config = RobotConfig()
        self.robot.impedance_control = FakeControl(self.robot.config.task)
        self.robot.gripper = FakeGripper()

    def test_only_yaw_speed_cap_is_raised(self):
        from shared.task.manipulation_robot import RobotConfig as BaseRobotConfig
        baseline = BaseRobotConfig(task=self.robot.config.task)
        changed = self.robot.config.impendence_control
        original = baseline.impendence_control
        np.testing.assert_allclose(changed.limit['max_speed'][0], original.limit['max_speed'][0])
        np.testing.assert_allclose(changed.limit['max_speed'][1, :2], original.limit['max_speed'][1, :2])
        self.assertAlmostEqual(changed.limit['max_speed'][1, 2], np.deg2rad(90.0))
        np.testing.assert_allclose(changed.limit['max_wrench'], original.limit['max_wrench'])

    def test_hold_cancels_stale_target_without_gripper_command(self):
        control = self.robot.impedance_control
        control.target_pose[0, 2] = .155
        self.robot.apply_action({'hold_current_pose': True})
        np.testing.assert_allclose(control.target_pose, control.pose)
        self.assertEqual(self.robot.gripper.commands, [])

    def test_round_trip_and_replay_batch(self):
        raw = {"delta_pos": [0.1, -0.2, 0.3], "delta_rot": [1, 1, 0.4], "gripper": 0.6}
        expected = [0.1, -0.2, 0.3, 0.4, 0.6]
        action = self.model.build_action(raw, {})
        np.testing.assert_allclose(action.numpy(), expected)
        np.testing.assert_allclose(self.model.build_actions([raw, raw], [{}, {}]), [expected, expected])
        parsed = self.model.parse_action(action)
        np.testing.assert_allclose(parsed["delta_rot"], [0, 0, 0.4])
        self.assertAlmostEqual(parsed["gripper"], 0.6, places=6)
        self.assertEqual(self.model.config.action_dims, 5)

    def test_human_gripper_buttons_and_local_yaw(self):
        action, intervention = self.model.select_action(
            torch.zeros(5), {"delta_rot": [0, 0, -0.3], "gripper_close_pressed": True}, -1.
        )
        self.assertTrue(intervention)
        np.testing.assert_allclose(action.numpy(), [0, 0, 0, -0.3, 1])
        self.assertNotIn("rotation_copilot", self.model.build_operator_request({"gripper": 0}))

    def test_xyz_local_yaw_gripper(self):
        self.robot.apply_action({"delta_pos": [1, -1, 1], "delta_rot": [1, 1, 1], "gripper": 1})
        task = self.robot.config.task
        target = self.robot.impedance_control.target_pose
        expected_position = np.clip(
            np.array(task.reset_pos) + [.05, -.05, .05],
            task.safety_pos_min,
            task.safety_pos_max,
        )
        np.testing.assert_allclose(target[0], expected_position)
        np.testing.assert_allclose(target[1], rotation_at_yaw(task, task.angular_speed), atol=1e-6)
        self.assertEqual(self.robot.gripper.commands[-1], (255, 128, 200))

    def test_accumulated_yaw_and_both_limits(self):
        task = self.robot.config.task
        for sign, bound in [(-1, task.yaw_range_rad[0]), (1, task.yaw_range_rad[1])]:
            for _ in range(int(np.ceil((task.yaw_range_rad[1] - task.yaw_range_rad[0]) / task.angular_speed)) + 2):
                self.robot.apply_action({"delta_rot": [0, 0, sign]})
            self.assertAlmostEqual(yaw_relative_to_nominal(task, self.robot.impedance_control.target_pose[1]), bound, places=5)

    def test_reset_target_projects_all_axes_and_gripper_speed(self):
        task = self.robot.config.task
        target = np.array([[1, 1, 1], rotation_at_yaw(task, 2.5)])
        self.robot.apply_action({"reset_target_pose": target, "gripper": -1})
        np.testing.assert_allclose(self.robot.impedance_control.target_pose[0], task.safety_pos_max)
        self.assertAlmostEqual(yaw_relative_to_nominal(task, self.robot.impedance_control.target_pose[1]), task.yaw_range_rad[1], places=5)
        self.assertEqual(self.robot.gripper.commands[-1], (0, 128, 200))

    def test_masked_reset_does_not_project_inactive_position_axes(self):
        task = self.robot.config.task
        actual = self.robot.impedance_control.pose
        actual[0] = np.asarray((-0.1184, -0.5405, 0.177), dtype=np.float32)
        requested = actual.copy()
        requested[0, 2] = task.reset_clearance_z
        requested[1] = rotation_at_yaw(task, math.radians(53))

        self.robot.apply_action(
            {
                "reset_target_pose": requested,
                "reset_position_axes": (False, False, True),
                "reset_rotation": True,
            }
        )

        target = self.robot.impedance_control.target_pose
        np.testing.assert_allclose(target[0, :2], actual[0, :2])
        self.assertAlmostEqual(target[0, 2], task.reset_clearance_z)
        self.assertAlmostEqual(
            yaw_relative_to_nominal(task, target[1]), math.radians(53), places=5
        )

    def test_gripper_only_and_recovery(self):
        original = self.robot.impedance_control.target_pose.copy()
        self.robot.apply_action({"gripper_only": True, "gripper": -1})
        np.testing.assert_allclose(self.robot.impedance_control.target_pose, original)
        self.assertEqual(self.robot.gripper.commands, [(0, 128, 200)])
        self.robot.impedance_control.recovering = True
        self.robot.apply_action({"delta_pos": [1, 1, 1], "gripper": 1})
        np.testing.assert_allclose(self.robot.impedance_control.target_pose, original)
        self.assertEqual(len(self.robot.gripper.commands), 1)

    def test_downward_command_respects_z_lower_bound(self):
        lower_z = self.robot.config.task.safety_pos_min[2]
        self.robot.impedance_control.pose[0, 2] = lower_z + 0.002
        self.robot.apply_action({"delta_pos": [0, 0, -1]})
        self.assertAlmostEqual(self.robot.impedance_control.target_pose[0, 2], lower_z)

    def test_reset_releases_retracts_returns_and_pauses(self):
        task = self.model.config.task
        initial_pos = np.asarray(task.reset_pos, dtype=np.float32).copy()
        initial_pos[2] = task.safety_pos_min[2]
        pose = np.array([initial_pos, rotation_at_yaw(task, .2)])
        commands = []

        def send(action):
            commands.append(action)
            if "reset_target_pose" in action:
                pose[:] = action["reset_target_pose"]

        offsets = [np.array([-.02, .02, 0.]), np.array([.02, -.02, 0.])]
        clock = [0.0]
        def advance(seconds):
            clock[0] += seconds

        with patch("tasks.rotate_knob.modeling.time.monotonic", side_effect=lambda: clock[0]), \
             patch("tasks.rotate_knob.modeling.time.sleep", side_effect=advance) as sleep, \
             patch("tasks.rotate_knob.modeling.np.random.uniform", side_effect=offsets) as sample:
            self.model.reset(send, lambda: {"tcp_pose": pose.copy()})
            np.testing.assert_allclose(pose[0], np.array(task.reset_pos) + offsets[0], atol=1e-6)
            self.assertFalse(any(c.args == (1.0,) for c in sleep.call_args_list))
            sleep.reset_mock()
            started = clock[0]
            self.model.reset(send, lambda: {"tcp_pose": pose.copy()})
            self.assertAlmostEqual(clock[0] - started, 7.0)
            self.assertEqual(sample.call_count, 2)
            for call in sample.call_args_list:
                spread = np.asarray(task.reset_rnd_abs)
                np.testing.assert_allclose(call.args[0], -spread)
                np.testing.assert_allclose(call.args[1], spread)
        self.assertTrue(commands[0]["gripper_only"])
        self.assertTrue(
            all(
                "gripper" not in command
                for command in commands
                if "reset_target_pose" in command
            )
        )
        reset_commands = [
            command for command in commands if "reset_target_pose" in command
        ]
        masks = [command["reset_position_axes"] for command in reset_commands]
        self.assertIn((False, False, False), masks)
        self.assertIn((False, False, True), masks)
        self.assertIn((True, True, False), masks)
        z_command = reset_commands[0]
        self.assertEqual(z_command["reset_position_axes"], (False, False, True))
        self.assertFalse(z_command["reset_rotation"])
        np.testing.assert_allclose(
            z_command["reset_target_pose"][0, :2], initial_pos[:2]
        )
        self.assertAlmostEqual(
            z_command["reset_target_pose"][0, 2], task.reset_clearance_z
        )

        rotation_commands = [
            command
            for command in reset_commands
            if command["reset_position_axes"] == (False, False, False)
        ]
        rotation_targets = [
            command["reset_target_pose"] for command in rotation_commands
        ]
        rotation_yaws = [
            yaw_relative_to_nominal(task, target[1])
            for target in rotation_targets
        ]
        self.assertGreater(len(rotation_yaws), 1)
        self.assertTrue(np.all(np.diff(rotation_yaws) > 0))
        self.assertAlmostEqual(rotation_yaws[-1], task.reset_yaw_rad, places=5)
        xy_commands = [
            command
            for command in reset_commands
            if command["reset_position_axes"] == (True, True, False)
        ]
        self.assertGreaterEqual(len(xy_commands), 1)
        np.testing.assert_allclose(
            xy_commands[0]["reset_target_pose"][0],
            np.asarray(task.reset_pos) + offsets[0],
        )
        np.testing.assert_allclose(pose[0], np.array(task.reset_pos) + offsets[1], atol=1e-6)
        self.assertAlmostEqual(
            yaw_relative_to_nominal(task, pose[1]), task.reset_yaw_rad, places=5
        )

    def test_reset_crosses_minus_100_to_plus_100_by_increasing_yaw(self):
        task = self.model.config.task
        pose = np.asarray(
            [task.reset_pos, rotation_at_yaw(task, math.radians(-100))],
            dtype=np.float32,
        )
        commanded_yaws = []

        def send(action):
            target = action.get("reset_target_pose")
            if target is not None:
                if np.allclose(target[0], task.reset_pos):
                    commanded_yaws.append(
                        yaw_relative_to_nominal(task, target[1])
                    )
                pose[:] = target

        with patch("tasks.rotate_knob.modeling.time.sleep"), patch(
            "tasks.rotate_knob.modeling.np.random.uniform",
            return_value=np.zeros(3),
        ):
            self.model.reset(send, lambda: {"tcp_pose": pose.copy()})

        self.assertGreater(len(commanded_yaws), 1)
        self.assertTrue(np.all(np.diff(commanded_yaws) > 0))
        self.assertAlmostEqual(
            commanded_yaws[0], math.radians(-70), places=5
        )
        self.assertAlmostEqual(commanded_yaws[-1], math.radians(100), places=5)

    def test_slow_robot_retains_configured_time_for_human(self):
        self.model._completed_resets = 1
        clock = [0.0]

        def advance(seconds):
            clock[0] += seconds

        def move(*args, **kwargs):
            del args, kwargs
            advance(3.0)
            return True

        with patch("tasks.rotate_knob.modeling.time.monotonic", side_effect=lambda: clock[0]), \
             patch("tasks.rotate_knob.modeling.time.sleep", side_effect=advance), \
             patch.object(self.model, "_move_reset_pose", side_effect=move) as move_reset:
            self.model.reset(Mock(), lambda: {"tcp_pose": self.robot.impedance_control.pose})
        self.assertAlmostEqual(
            clock[0],
            move_reset.call_count * 3.0 + self.model.config.task.human_reset_seconds,
        )

    def test_rotation_reset_timeout_does_not_proceed(self):
        pose = self.robot.impedance_control.pose
        send = Mock()
        with patch.object(
            self.model, "_rotate_to_reset_yaw_increasing", return_value=False
        ) as rotate:
            with self.assertRaisesRegex(RuntimeError, "rotation reset timed out"):
                self.model.reset(send, lambda: {"tcp_pose": pose})
        self.assertEqual(rotate.call_count, 1)
        send.assert_called_with({'hold_current_pose': True})

    def test_z_stage_timeout_cancels_target(self):
        send = Mock()
        with patch.object(
            self.model, "_rotate_to_reset_yaw_increasing", return_value=True
        ), patch.object(self.model, "_move_reset_pose", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Z reset timed out"):
                self.model.reset(
                    send, lambda: {"tcp_pose": self.robot.impedance_control.pose}
                )
        send.assert_called_with({'hold_current_pose': True})
        self.assertFalse(hasattr(self.model, '_completed_resets'))


if __name__ == "__main__":
    unittest.main()
