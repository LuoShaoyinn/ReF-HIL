from __future__ import annotations

import unittest
from dataclasses import fields

from tasks.insert_usb.config import TaskConfig as InsertUSBTaskConfig

from .config import TaskConfig
from .robot import RobotConfig


class InsertRJ45ConfigTest(unittest.TestCase):
    def test_all_parameters_match_insert_usb_except_name(self) -> None:
        rj45 = TaskConfig()
        usb = InsertUSBTaskConfig()
        for field in fields(TaskConfig):
            if field.name == "name":
                continue
            self.assertEqual(
                getattr(rj45, field.name),
                getattr(usb, field.name),
                field.name,
            )
        self.assertEqual(rj45.name, "insert_rj45")

    def test_robot_sdk_uses_maximum_gripper_settings(self) -> None:
        config = RobotConfig()
        self.assertEqual(config.gripper.runtime_speed_limit, 255)
        self.assertEqual(config.gripper.runtime_force_limit, 255)


if __name__ == "__main__":
    unittest.main()
