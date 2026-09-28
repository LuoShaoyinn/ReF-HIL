from __future__ import annotations

import unittest

from shared.task.manipulation_config import TaskConfig as BaseTaskConfig

from .config import TaskConfig
from .robot import RobotConfig


class InsertUSBConfigTest(unittest.TestCase):
    def test_action_and_hardware_contract_matches_shared_defaults(self) -> None:
        config = TaskConfig()
        defaults = BaseTaskConfig()
        self.assertEqual(config.name, "insert_usb")
        self.assertFalse(config.allow_rotation)
        self.assertEqual(config.action_dims, 4)
        self.assertEqual(config.max_episode_steps, 200)
        self.assertAlmostEqual(config.step_penalty, 0.005)
        self.assertEqual(config.linear_speed, defaults.linear_speed)
        self.assertEqual(
            config.gripper_position_range,
            defaults.gripper_position_range,
        )
        self.assertEqual(config.gripper_speed, 255)
        self.assertEqual(config.gripper_force, 255)
        self.assertEqual(config.zero_point_rot, defaults.zero_point_rot)

    def test_calibrated_workspace_reset_and_classifier_crop(self) -> None:
        config = TaskConfig()
        self.assertEqual(config.safety_pos_min, (-0.175, -0.700, -0.007))
        self.assertEqual(config.safety_pos_max, (-0.040, -0.425, 0.100))
        self.assertEqual(config.reset_pos, (-0.150, -0.475, 0.040))
        self.assertTrue(
            all(
                lower <= value <= upper
                for value, lower, upper in zip(
                    config.reset_pos,
                    config.safety_pos_min,
                    config.safety_pos_max,
                    strict=True,
                )
            )
        )
        self.assertEqual(config.success_conditions[0].clip, (261, 200, 56, 56))

    def test_robot_sdk_limits_follow_task_gripper_settings(self) -> None:
        config = RobotConfig()
        self.assertEqual(config.gripper.runtime_speed_limit, 255)
        self.assertEqual(config.gripper.runtime_force_limit, 255)


if __name__ == "__main__":
    unittest.main()
