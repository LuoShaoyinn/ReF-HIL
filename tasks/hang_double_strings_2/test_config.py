"""Configuration contract for the calibrated paper double-string task."""

import unittest

from tasks import load_component
from .config import TaskConfig


class HangDoubleStringsConfigTest(unittest.TestCase):
    def test_workspace_reset_and_camera_contract(self):
        config = TaskConfig()
        self.assertEqual(config.name, "hang_double_strings_2")
        self.assertEqual(config.safety_pos_min, (-0.320, -0.700, -0.004))
        self.assertEqual(config.safety_pos_max, (-0.200, -0.450, 0.060))
        self.assertEqual(config.reset_pos, (-0.300, -0.600, 0.060))
        self.assertEqual(config.reset_rnd_abs, (0.0, 0.040, 0.0))
        self.assertEqual(config.global_camera_resolution, (1280, 720))
        self.assertEqual(config.global_policy_clip, (350, 94, 448, 448))
        self.assertEqual(config.wrist_0_policy_clip, (0, 0, 448, 448))
        self.assertEqual(config.wrist_1_policy_clip, (0, 0, 448, 448))

    def test_two_classifier_contract(self):
        config = TaskConfig()
        self.assertEqual(
            tuple(condition.name for condition in config.success_conditions),
            ("upper_clip", "lower_clip"),
        )
        self.assertEqual(config.success_conditions[0].clip, (440, 328, 80, 80))
        self.assertEqual(config.success_conditions[1].clip, (615, 201, 64, 64))
        self.assertEqual(
            tuple(condition.checkpoint for condition in config.success_conditions),
            ("classifiers/upper_clip.pt", "classifiers/lower_clip.pt"),
        )

    def test_dynamic_bindings_without_hardware_initialization(self):
        for component in ("modeling", "classifier"):
            module = load_component("hang_double_strings_2", component)
            config = getattr(module, component.title() + "Config")(device="cpu")
            self.assertEqual(config.task, TaskConfig())
        module = load_component("hang_double_strings_2", "learner")
        from learner.training import LimitActionLearnerConfig

        self.assertTrue(issubclass(module.LearnerConfig, LimitActionLearnerConfig))


if __name__ == '__main__':
    unittest.main()
