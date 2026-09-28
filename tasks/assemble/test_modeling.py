from __future__ import annotations

import unittest

import numpy as np

from .modeling import Modeling, ModelingConfig


class AssembleModelingTest(unittest.TestCase):
    def test_prepare_observation_normalizes_gripper_and_omits_pose(self) -> None:
        modeling = object.__new__(Modeling)
        modeling.config = ModelingConfig(device="cpu")
        raw = {
            "tcp_pose": np.zeros((2, 3), dtype=np.float32),
            "tcp_speed": np.zeros((2, 3), dtype=np.float32),
            "tcp_force": np.zeros((2, 3), dtype=np.float32),
            "gripper": np.float32(32.0),
            "images": {"global_0": np.zeros((2, 2, 3), dtype=np.uint8)},
        }
        prepared = modeling.prepare_observation(raw)
        self.assertNotIn("tcp_pose", prepared)
        self.assertAlmostEqual(float(prepared["gripper"]), 0.0)
        self.assertEqual(prepared["tcp_speed"].shape, (6,))
        self.assertEqual(prepared["tcp_force"].shape, (6,))
        self.assertEqual(prepared["projected_gravity"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
