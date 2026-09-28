from __future__ import annotations

import unittest

import numpy as np

from .reset import ResetConfig, run_reset


class AssembleResetTest(unittest.TestCase):
    def test_reset_opens_then_only_lifts_and_moves(self) -> None:
        config = ResetConfig(
            poll_interval_s=0.0,
            move_timeout_s=1.0,
            move_tolerance=1e-5,
        )
        position = np.asarray([-0.162, -0.586, 0.039], dtype=np.float32)
        actions: list[dict] = []

        def read_observation() -> dict:
            return {
                "tcp_pose": np.asarray(
                    [position, np.zeros(3, dtype=np.float32)], dtype=np.float32
                )
            }

        def send_action(action: dict) -> None:
            actions.append(action)
            position[:] += (
                np.asarray(action["delta_pos"], dtype=np.float32)
                * config.linear_speed
            )

        target = run_reset(
            send_action,
            read_observation,
            config=config,
            rng=np.random.default_rng(7),
        )

        self.assertTrue(actions)
        self.assertEqual(float(actions[0]["gripper"]), -1.0)
        np.testing.assert_allclose(actions[0]["delta_pos"], 0.0)
        self.assertTrue(all("gripper" not in action for action in actions[1:]))
        np.testing.assert_allclose(actions[1]["delta_pos"][:2], 0.0)
        self.assertGreater(float(actions[1]["delta_pos"][2]), 0.0)
        self.assertAlmostEqual(float(target[2]), config.travel_z)
        np.testing.assert_allclose(position, target, atol=config.move_tolerance)


if __name__ == "__main__":
    unittest.main()
