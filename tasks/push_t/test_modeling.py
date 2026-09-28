from __future__ import annotations

import unittest

import numpy as np
import torch

from .config import TaskConfig
from .modeling import Modeling, ModelingConfig


class PushTModelingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.modeling = object.__new__(Modeling)
        self.modeling.config = ModelingConfig(device="cpu")
        self.modeling.device = torch.device("cpu")

    def test_full_timeout_return_is_negative_one(self) -> None:
        task = self.modeling.config.task
        self.assertAlmostEqual(
            task.max_episode_steps * task.step_penalty,
            1.0,
            places=7,
        )

    def test_policy_action_is_xyz_translation_only(self) -> None:
        action = self.modeling.build_action(
            {
                "delta_pos": [0.25, -0.5, 0.75],
                "delta_rot": [0.9, -0.8, 0.4],
                "gripper": -1.0,
            },
            {},
        )

        np.testing.assert_allclose(action.numpy(), [0.25, -0.5, 0.75])

    def test_policy_action_expands_to_fixed_rotation_and_closed_gripper(self) -> None:
        raw = self.modeling.parse_action(
            torch.tensor([0.2, -0.3, 0.4], dtype=torch.float32)
        )

        np.testing.assert_allclose(raw["delta_pos"], [0.2, -0.3, 0.4])
        np.testing.assert_allclose(raw["delta_rot"], [0.0, 0.0, 0.0])
        self.assertEqual(raw["gripper"], 1.0)

    def test_any_manual_axis_or_button_selects_human_action(self) -> None:
        policy = torch.zeros(3)
        action, intervened = self.modeling.select_action(
            policy,
            {
                "delta_pos": [0.0, 0.0, 0.0],
                "delta_rot": [1.0, 1.0, 1.0],
                "gripper_pressed": True,
                "gripper": -1.0,
                "is_intervene": False,
            },
            1.0,
        )
        self.assertTrue(intervened)
        np.testing.assert_allclose(action.numpy(), [0.0, 0.0, 0.0])

        action, intervened = self.modeling.select_action(
            policy,
            {
                "delta_pos": [0.0, 0.0, 0.5],
                "delta_rot": [0.0, 0.0, 0.0],
                "is_intervene": True,
            },
            1.0,
        )
        self.assertTrue(intervened)
        np.testing.assert_allclose(action.numpy(), [0.0, 0.0, 0.5])

    def test_all_manual_axes_and_buttons_start_demo_recording(self) -> None:
        self.assertTrue(self.modeling.operator_has_intent({
            "delta_pos": [0.0, 0.0, 0.0],
            "delta_rot": [1.0, 1.0, 0.0],
            "gripper_pressed": True,
        }))
        self.assertTrue(self.modeling.operator_has_intent({
            "delta_pos": [0.0, 0.0, 0.2],
            "delta_rot": [0.0, 0.0, 0.0],
        }))

    def test_reset_moves_y_before_lifting_and_x(self) -> None:
        self.modeling.config = ModelingConfig(
            device="cpu",
            task=TaskConfig(
                reset_step_duration_s=0.0,
                reset_settle_time_s=0.0,
            ),
        )
        task = self.modeling.config.task
        pose = np.asarray(
            [[task.reset_pos[0], -0.550, 0.040], task.zero_point_rot],
            dtype=np.float32,
        )
        commanded: list[np.ndarray] = []

        def read_observation() -> dict:
            return {"tcp_pose": pose.copy()}

        def send_action(action: dict) -> None:
            target = np.asarray(
                action["reset_target_pose"], dtype=np.float32
            ).reshape(2, 3)
            commanded.append(target.copy())
            pose[:] = target

        self.modeling.reset(send_action, read_observation)

        self.assertGreaterEqual(len(commanded), 2)
        self.assertAlmostEqual(commanded[0][0, 1], -0.400, places=6)
        self.assertAlmostEqual(commanded[0][0, 2], 0.040, places=6)
        self.assertAlmostEqual(commanded[1][0, 1], -0.400, places=6)
        self.assertAlmostEqual(commanded[1][0, 2], 0.060, places=6)
        self.assertAlmostEqual(commanded[-1][0, 1], -0.400, places=6)
        self.assertAlmostEqual(commanded[-1][0, 2], 0.060, places=6)
        np.testing.assert_allclose(commanded[-1][1], task.zero_point_rot)


if __name__ == "__main__":
    unittest.main()
