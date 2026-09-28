from __future__ import annotations

import unittest

import numpy as np
import torch

from .modeling import Modeling, ModelingConfig, rotation_at_yaw, yaw_relative_to_nominal


class GearModelingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.modeling = object.__new__(Modeling)
        self.modeling.config = ModelingConfig(device="cpu")
        self.modeling.device = torch.device("cpu")

    def test_action_contract_is_xyz_local_yaw_and_gripper(self) -> None:
        raw = {
            "delta_pos": np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
            "delta_rot": np.asarray([0.8, -0.7, 0.4], dtype=np.float32),
            "gripper": 0.6,
        }
        action = self.modeling.build_action(raw, {})
        np.testing.assert_allclose(action.numpy(), [0.1, -0.2, 0.3, 0.4, 0.6])
        parsed = self.modeling.parse_action(action)
        np.testing.assert_allclose(parsed["delta_pos"], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(parsed["delta_rot"], [0.0, 0.0, 0.4])
        self.assertAlmostEqual(parsed["gripper"], 0.6, places=6)

    def test_yaw_round_trip_across_task_range(self) -> None:
        task = self.modeling.config.task
        for yaw in (0.0, -0.5, -1.0, -np.pi + 1e-4):
            recovered = yaw_relative_to_nominal(task, rotation_at_yaw(task, yaw))
            self.assertAlmostEqual(recovered, yaw, places=5)

    def test_reset_opens_then_lifts_then_moves_to_random_target(self) -> None:
        task = self.modeling.config.task
        pose = np.asarray(
            [[-0.15, -0.50, 0.06], rotation_at_yaw(task, 0.4)],
            dtype=np.float32,
        )
        commands: list[dict] = []

        def read_observation() -> dict:
            return {"tcp_pose": pose.copy()}

        def send_action(action: dict) -> None:
            commands.append(action)
            if "reset_target_pose" in action:
                pose[:] = np.asarray(action["reset_target_pose"], dtype=np.float32)

        np.random.seed(7)
        self.modeling.reset(send_action, read_observation)

        self.assertGreaterEqual(len(commands), 3)
        self.assertEqual(commands[0]["gripper"], -1.0)
        self.assertTrue(commands[0]["gripper_only"])
        self.assertNotIn("reset_target_pose", commands[0])
        self.assertAlmostEqual(commands[1]["reset_target_pose"][0, 0], -0.15)
        self.assertAlmostEqual(commands[1]["reset_target_pose"][0, 1], -0.50)
        self.assertAlmostEqual(commands[1]["reset_target_pose"][0, 2], 0.110)
        final_pose = np.asarray(commands[-1]["reset_target_pose"])
        self.assertTrue(-0.250 <= final_pose[0, 0] <= -0.150)
        self.assertTrue(-0.470 <= final_pose[0, 1] <= -0.430)
        self.assertAlmostEqual(final_pose[0, 2], 0.110)
        self.assertAlmostEqual(yaw_relative_to_nominal(task, final_pose[1]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
