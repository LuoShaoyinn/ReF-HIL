from __future__ import annotations

import unittest

from .config import TaskConfig
from .robot import RobotConfig


class PlugInConfigTest(unittest.TestCase):
    def test_action_workspace_and_reward_contract(self) -> None:
        config = TaskConfig()
        self.assertEqual(config.name, "plug_in")
        self.assertEqual(config.action_dims, 4)
        self.assertFalse(config.allow_rotation)
        self.assertEqual(config.max_episode_steps, 200)
        self.assertAlmostEqual(config.step_penalty, 0.005)
        self.assertEqual(config.safety_pos_min, (-0.250, -0.720, 0.030))
        self.assertEqual(config.safety_pos_max, (-0.150, -0.470, 0.110))
        self.assertEqual(config.reset_pos, (-0.200, -0.595, 0.100))
        self.assertEqual(config.reset_rnd_abs, (0.050, 0.125, 0.0))
        self.assertEqual(config.reset_move_timeout_s, 2.0)
        self.assertEqual(config.reset_settle_time_s, 3.0)
        self.assertEqual(config.success_conditions[0].clip, (292, 108, 112, 112))
        self.assertEqual(config.global_policy_clip, (158, 86, 224, 224))
        self.assertEqual(config.wrist_0_policy_clip, (80, 0, 480, 480))
        self.assertEqual(config.wrist_1_policy_clip, (80, 0, 480, 480))

    def test_gripper_uses_minimum_speed_and_maximum_force(self) -> None:
        config = TaskConfig()
        robot = RobotConfig()
        self.assertEqual(config.gripper_speed, 0)
        self.assertEqual(config.gripper_force, 255)
        self.assertEqual(robot.gripper.runtime_speed_limit, 0)
        self.assertEqual(robot.gripper.runtime_force_limit, 255)


if __name__ == "__main__":
    unittest.main()
