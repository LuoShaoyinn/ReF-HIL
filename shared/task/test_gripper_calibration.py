"""Readback/action round trips without hardware or model downloads."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from actor.operator.toggle import resolve_gripper_command
from actor.robot.action import normalized_gripper_position
from actor.transport import FakeRobotClient
from .gripper_calibration import gripper_control_position
from .manipulation_modeling import Modeling, ModelingConfig

for name in ("rtde_control", "rtde_receive", "pyrealsense2"):
    sys.modules.setdefault(name, ModuleType(name))

from .manipulation_robot import Robot, RobotConfig
from tasks.assemble.robot import Robot as AssembleRobot, RobotConfig as AssembleRobotConfig


class GripperCalibrationTest(unittest.TestCase):
    def test_reproduce_previous_248_to_206_bug(self):
        position, last = 248, 1.
        sequence = [position]
        for _ in range(8):
            command = resolve_gripper_command([0, 0], position / 255, last_command=last)
            position = normalized_gripper_position(command, 3, 248)
            sequence.append(position)
            last = command
        self.assertEqual(sequence, [248, 241, 235, 229, 223, 217, 211, 206, 206])

    def test_calibrated_release_never_drifts(self):
        for opened, closed in ((3, 248), (0, 255), (10, 210), (3, 64)):
            for start in (opened, (opened + closed) // 2, closed):
                position = start
                last = 2 * (start - opened) / (closed - opened) - 1
                for _ in range(100):
                    feedback = gripper_control_position(
                        {"gripper": position, "gripper_action_range": (opened, closed)}, (0, 255)
                    )
                    last = resolve_gripper_command([0, 0], feedback, last_command=last)
                    position = normalized_gripper_position(last, opened, closed)
                    self.assertEqual(position, start)

    def test_legacy_fallback_and_clamping(self):
        self.assertEqual(gripper_control_position({"gripper": 32}, (0, 64)), .5)
        self.assertEqual(gripper_control_position({"gripper": -1, "gripper_action_range": (3,248)}, (0,255)), 0)
        self.assertEqual(gripper_control_position({"gripper": 255, "gripper_action_range": (3,248)}, (0,255)), 1)

    def test_invalid_metadata_does_not_fall_back(self):
        for endpoints in (None, (3,3), (248,3), (0,float('nan')), (1,2,3)):
            with self.assertRaises(ValueError):
                gripper_control_position({"gripper": 248, "gripper_action_range": endpoints}, (0,255))
        with self.assertRaises(ValueError):
            gripper_control_position({"gripper": float('nan')}, (0,255))

    def test_actor_demo_operator_request_uses_live_endpoints(self):
        model = object.__new__(Modeling)
        model.config = ModelingConfig(device="cpu")
        raw = {"tcp_pose": np.zeros((2,3)), "gripper": 248, "gripper_action_range": (3,248)}
        self.assertEqual(model.build_operator_request(raw)["gripper_position"], 1.)
        raw.update(gripper=3)
        self.assertEqual(model.build_gripper_operator_request(raw)["gripper_position"], 0.)

    def test_saved_policy_observation_is_unchanged(self):
        model = object.__new__(Modeling)
        model.config = ModelingConfig(device="cpu")
        raw = {"tcp_pose": np.zeros((2,3)), "tcp_speed": np.zeros(6), "tcp_force": np.zeros(6),
               "gripper": 248, "images": {}}
        legacy = model.prepare_observation(raw)
        raw['gripper_action_range'] = (3,248)
        updated = model.prepare_observation(raw)
        self.assertEqual(legacy['gripper'], updated['gripper'])
        self.assertNotIn('gripper_action_range', updated)

    def test_robot_observation_emits_current_calibration_without_motion(self):
        robot = object.__new__(Robot)
        robot.config = RobotConfig()
        robot.gripper = Mock()
        robot.gripper.get_open_position.return_value = 3
        robot.gripper.get_closed_position.return_value = 248
        robot.gripper.get_current_position.return_value = 248
        robot.impedance_control = Mock()
        for method in ('get_actual_tcp_pose', 'get_actual_tcp_speed', 'get_actual_tcp_force'):
            getattr(robot.impedance_control, method).return_value = np.zeros((2,3))
        camera = SimpleNamespace(read_image=lambda: np.zeros((480,640,3), dtype=np.uint8),
                                 read_depth=lambda: np.zeros((480,640), dtype=np.uint8))
        robot.cameras = {key: camera for key in ('global', 'wrist_0', 'wrist_1')}
        raw = robot.read_obs()
        self.assertEqual(raw['gripper_action_range'], (3,248))
        self.assertEqual(gripper_control_position(raw, (0,255)), 1.)
        robot.gripper.move.assert_not_called()
        robot.gripper.activate.assert_not_called()

    def test_assemble_reports_effective_partial_travel(self):
        robot = object.__new__(AssembleRobot)
        robot.config = AssembleRobotConfig()
        robot.gripper = Mock()
        robot.gripper.get_open_position.return_value = 3
        robot.gripper.get_closed_position.return_value = 248
        self.assertEqual(robot.gripper_action_range(), (3,64))

    def test_fake_robot_reports_its_action_range(self):
        robot = FakeRobotClient(image_size=2, linear_speed=.05, initial_position=(0,0,0),
                                initial_rotation=(0,0,0), gripper_position_range=(3,248))
        robot.send_action({'gripper': 1})
        obs = robot.read_observation()
        self.assertEqual(obs['gripper_action_range'], (3,248))
        self.assertEqual(gripper_control_position(obs, (0,255)), 1.)


if __name__ == '__main__':
    unittest.main()
