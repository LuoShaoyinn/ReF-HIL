from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from actor.operator.copilot import rotation_copilot_action


class RotationCopilotTest(unittest.TestCase):
    def test_returns_local_shortest_path_matching_robot_post_multiply(self) -> None:
        current = R.from_euler("z", 90.0, degrees=True)
        local_step = R.from_euler("x", 5.0, degrees=True)
        target = current * local_step

        action = rotation_copilot_action(
            current.as_rotvec(),
            target.as_rotvec(),
            angular_speed=np.deg2rad(10.0),
            deadband_rad=0.0,
        )

        np.testing.assert_allclose(action, [0.5, 0.0, 0.0], atol=1e-6)

    def test_large_error_is_clipped_to_normalized_action_range(self) -> None:
        action = rotation_copilot_action(
            np.zeros(3),
            np.asarray([0.5, -0.5, 0.5]),
            angular_speed=0.1,
            deadband_rad=0.0,
        )

        self.assertTrue(np.all(np.abs(action) <= 1.0))
        self.assertTrue(np.any(np.abs(action) == 1.0))

    def test_deadband_outputs_zero(self) -> None:
        action = rotation_copilot_action(
            np.zeros(3),
            np.asarray([0.001, 0.0, 0.0]),
            angular_speed=0.1,
            deadband_rad=0.01,
        )

        np.testing.assert_array_equal(action, np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
