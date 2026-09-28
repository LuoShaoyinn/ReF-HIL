from __future__ import annotations

import math
import unittest

from learner.training import LimitActionLearnerConfig
from shared.task.manipulation_config import TaskConfig as BaseTaskConfig
from tasks import load_component


class DynamicTaskLoadingTest(unittest.TestCase):
    def test_components_resolve_by_package_name_without_registry(self) -> None:
        for task_name in (
            "hang_double_strings_2",
            "assemble",
            "insert_usb",
            "insert_rj45",
            "plug_in",
            "plug_power_socket",
            "push_t",
            "gear",
            "insert_gear",
            "rotate_knob",
        ):
            config = load_component(task_name, "config").TaskConfig()
            modeling_config = load_component(task_name, "modeling").ModelingConfig(
                device="cpu"
            )
            learner_config_type = load_component(task_name, "learner").LearnerConfig
            self.assertEqual(config.name, task_name)
            self.assertEqual(modeling_config.action_dims, config.action_dims)
            self.assertTrue(issubclass(learner_config_type, LimitActionLearnerConfig))

    def test_double_string_task_requires_both_clip_conditions(self) -> None:
        config = load_component("hang_double_strings_2", "config").TaskConfig()
        self.assertEqual(
            tuple(condition.name for condition in config.success_conditions),
            ("upper_clip", "lower_clip"),
        )
        self.assertEqual(config.action_dims, 4)
        self.assertFalse(config.allow_rotation)
        self.assertEqual(config.max_episode_steps, 200)
        self.assertEqual(config.safety_pos_min, (-0.320, -0.700, -0.004))
        self.assertEqual(config.safety_pos_max, (-0.200, -0.450, 0.060))

    def test_assemble_is_translation_plus_continuous_gripper(self) -> None:
        config = load_component("assemble", "config").TaskConfig()
        self.assertFalse(config.allow_rotation)
        self.assertEqual(config.action_dims, 4)
        self.assertEqual(config.max_episode_steps, 200)
        self.assertAlmostEqual(config.step_penalty, 0.005)
        self.assertAlmostEqual(
            -config.max_episode_steps * config.step_penalty,
            -1.0,
        )
        self.assertEqual(config.gripper_openness_range, (0.75, 1.0))
        self.assertEqual(config.safety_pos_min, (-0.250, -0.650, 0.005))
        self.assertEqual(config.safety_pos_max, (-0.080, -0.400, 0.120))
        self.assertEqual(
            config.movable_axes,
            (True, True, True, False, False, False),
        )

    def test_insert_usb_uses_calibrated_workspace_and_hang_string_controls(
        self,
    ) -> None:
        config = load_component("insert_usb", "config").TaskConfig()
        defaults = BaseTaskConfig()
        self.assertFalse(config.allow_rotation)
        self.assertEqual(config.action_dims, 4)
        self.assertEqual(config.linear_speed, defaults.linear_speed)
        self.assertEqual(
            config.gripper_position_range,
            defaults.gripper_position_range,
        )
        self.assertEqual(config.zero_point_rot, defaults.zero_point_rot)
        self.assertEqual(config.safety_pos_min, (-0.175, -0.700, -0.007))
        self.assertEqual(config.safety_pos_max, (-0.040, -0.425, 0.100))
        self.assertEqual(config.reset_pos, (-0.150, -0.475, 0.040))
        self.assertEqual(config.success_conditions[0].clip, (261, 200, 56, 56))

    def test_push_t_is_xyz_translation_with_fixed_orientation(self) -> None:
        config = load_component("push_t", "config").TaskConfig()
        self.assertEqual(config.action_dims, 3)
        self.assertEqual(config.safety_pos_min, (-0.300, -0.630, 0.025))
        self.assertEqual(config.safety_pos_max, (-0.080, -0.400, 0.060))
        self.assertEqual(config.reset_pos, (-0.190, -0.400, 0.060))
        self.assertEqual(config.reset_rnd_abs, (0.110, 0.0, 0.0))
        self.assertEqual(config.fixed_gripper_action, 1.0)
        self.assertEqual(config.global_camera_resolution, (1280, 720))
        self.assertEqual(config.success_conditions[0].clip, (296, 300, 228, 228))
        self.assertEqual(config.global_policy_clip, (300, 136, 640, 400))
        self.assertEqual(config.wrist_0_policy_clip, (80, 0, 480, 480))
        self.assertEqual(config.wrist_1_policy_clip, (80, 0, 480, 480))
        self.assertEqual(
            config.movable_axes,
            (True, True, True, False, False, False),
        )

    def test_gear_uses_xyz_yaw_and_continuous_gripper(self) -> None:
        config = load_component("gear", "config").TaskConfig()
        self.assertTrue(config.allow_rotation)
        self.assertEqual(config.action_dims, 5)
        self.assertEqual(config.safety_pos_min, (-0.300, -0.620, 0.040))
        self.assertEqual(config.safety_pos_max, (-0.110, -0.420, 0.150))
        self.assertEqual(config.yaw_range_rad, (-math.pi, 0.0))
        self.assertEqual(config.reset_pos, (-0.200, -0.450, 0.110))
        self.assertEqual(config.reset_rnd_abs, (0.050, 0.020, 0.0))
        self.assertEqual(config.reset_gripper, -1.0)
        self.assertEqual(config.reset_stage_timeout_s, 2.0)
        self.assertEqual(config.success_conditions[0].clip, (33, 142, 112, 112))
        self.assertEqual(config.global_policy_clip, (170, 10, 448, 448))
        self.assertEqual(config.wrist_0_policy_clip, (80, 0, 480, 480))
        self.assertEqual(config.wrist_1_policy_clip, (80, 0, 480, 480))
        self.assertEqual(
            config.movable_axes,
            (True, True, True, False, False, True),
        )

    def test_insert_gear_uses_xyz_yaw_and_fixed_closed_gripper(self) -> None:
        config = load_component("insert_gear", "config").TaskConfig()
        self.assertTrue(config.allow_rotation)
        self.assertEqual(config.action_dims, 4)
        self.assertEqual(config.max_episode_steps, 150)
        self.assertAlmostEqual(config.step_penalty, 1.0 / 150.0)
        self.assertEqual(config.safety_pos_min, (-0.1755, -0.5257, 0.033))
        self.assertEqual(config.safety_pos_max, (0.0245, -0.3507, 0.121))
        self.assertEqual(config.yaw_range_rad, (-math.pi / 2.0, math.pi / 9.0))
        self.assertEqual(config.yaw_speed_limit_rad_s, math.radians(90))
        self.assertEqual(config.angular_speed, math.radians(9))
        self.assertEqual(config.reset_yaw_waypoint_step_rad, math.radians(30))
        self.assertEqual(config.reset_pos, (-0.0755, -0.4382, 0.121))
        self.assertEqual(config.reset_rnd_abs, (0.1000, 0.0875, 0.0))
        self.assertEqual(config.reset_settle_time_s, 1.0)
        self.assertEqual(config.startup_pickup_pos, (-0.1180, -0.4600, 0.035))
        self.assertEqual(config.reset_release_z, 0.048)
        self.assertEqual(config.fixed_gripper_action, 1.0)
        self.assertEqual(config.gripper_speed, 0)
        self.assertEqual(config.gripper_force, 255)
        self.assertEqual(config.global_camera_resolution, (1280, 720))
        self.assertEqual(config.global_policy_clip, (663, 138, 294, 294))
        self.assertEqual(
            tuple(condition.name for condition in config.success_conditions),
            ("classifier_1", "classifier_2"),
        )
        self.assertEqual(config.success_conditions[0].clip, (685, 299, 80, 82))
        self.assertEqual(config.success_conditions[1].clip, (760, 304, 67, 67))
        self.assertEqual(
            tuple(condition.checkpoint for condition in config.success_conditions),
            ("classifiers/classifier_1.pt", "classifiers/classifier_2.pt"),
        )

    def test_unknown_task_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not provide component"):
            load_component("missing_task", "modeling")

    def test_task_name_cannot_escape_package(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid task name"):
            load_component("../learner", "modeling")


if __name__ == "__main__":
    unittest.main()
