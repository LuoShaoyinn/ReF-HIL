from dataclasses import replace
import math
import unittest

from tasks import load_component
from tasks.hang_double_strings_2.config import TaskConfig as HangConfig
from .config import TaskConfig


class KnobConfigTest(unittest.TestCase):
    def test_user_selected_limits(self):
        task = TaskConfig()
        self.assertEqual(task.yaw_range_rad, (math.radians(-100), math.radians(100)))
        self.assertEqual(task.safety_pos_min, (-0.204, -0.601, 0.109))
        self.assertEqual(task.safety_pos_max, (-0.044, -0.451, 0.160))
        self.assertEqual(task.reset_pos[2], 0.160)
        self.assertEqual(task.reset_clearance_z, task.reset_pos[2])
        self.assertEqual(task.reset_pos, (-0.124, -0.526, 0.160))
        self.assertEqual(task.reset_rnd_abs, (0.080, 0.075, 0.0))
        self.assertEqual(task.yaw_speed_limit_rad_s, math.radians(90))
        self.assertEqual(task.angular_speed, math.radians(9))
        self.assertEqual(task.reset_yaw_waypoint_step_rad, math.radians(30))
        self.assertEqual(task.reset_stage_timeout_s, 5.0)
        self.assertEqual(task.human_reset_seconds, 5)
        self.assertEqual(task.global_camera_resolution, (1280, 720))
        self.assertEqual(task.global_policy_clip, (618, 1, 300, 300))
        self.assertEqual(task.wrist_0_policy_clip, (0, 0, 448, 448))
        self.assertEqual(task.wrist_1_policy_clip, (0, 0, 448, 448))
        self.assertEqual(task.success_conditions[0].clip, (700, 110, 100, 100))

    def test_discovery_and_action_shape(self):
        task = load_component("rotate_knob", "config").TaskConfig()
        self.assertEqual(task.action_dims, 5)
        self.assertEqual(task.movable_axes, (True, True, True, False, False, True))
        self.assertTrue(task.allow_rotation)

    def test_gripper_matches_double_string_task(self):
        knob, hang = TaskConfig(), HangConfig()
        for key in ("gripper_speed", "gripper_position_range",
                    "reset_gripper", "gripper_reversal_penalty", "gripper_direction_deadband"):
            self.assertEqual(getattr(knob, key), getattr(hang, key), key)
        self.assertEqual(knob.gripper_force, 200)

    def test_single_independent_classifier(self):
        conditions = TaskConfig().success_conditions
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].checkpoint, "classifier.pt")
        self.assertEqual(conditions[0].image_key, "classifier")

    def test_invalid_reset_and_yaw_rejected(self):
        for overrides in ({"yaw_range_rad": (1., -1.)}, {"reset_yaw_rad": 2.},
                          {"reset_clearance_z": 2.}, {"reset_pos": (1., 1., 1.)}):
            with self.assertRaises(ValueError):
                replace(TaskConfig(), **overrides)


if __name__ == "__main__":
    unittest.main()
