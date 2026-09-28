from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from actor.robot.rotation import (
    apply_local_rotation_delta,
    clip_rotation_relative_to_nominal,
)


class LocalFrameRotationTest(unittest.TestCase):
    def test_delta_is_post_multiplied_in_gripper_frame(self) -> None:
        actual = R.from_euler("z", 90.0, degrees=True)
        delta = R.from_euler("x", 10.0, degrees=True)

        result = apply_local_rotation_delta(
            actual.as_rotvec(), delta.as_rotvec()
        )

        np.testing.assert_allclose(
            R.from_rotvec(result).as_matrix(),
            (actual * delta).as_matrix(),
            atol=1e-10,
        )
        self.assertFalse(
            np.allclose(
                R.from_rotvec(result).as_matrix(),
                (delta * actual).as_matrix(),
            )
        )

    def test_relative_safety_does_not_stick_at_nominal_pi_branch(self) -> None:
        nominal = np.asarray([2.221, 2.221, 0.0])
        half_range = np.deg2rad([45.0, 45.0, 90.0])

        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                delta = np.zeros(3)
                delta[axis] = sign * 0.02
                candidate = apply_local_rotation_delta(nominal, delta)
                safe = clip_rotation_relative_to_nominal(
                    candidate, nominal, half_range
                )
                applied_delta = (
                    R.from_rotvec(nominal).inv() * R.from_rotvec(safe)
                ).magnitude()
                self.assertAlmostEqual(applied_delta, 0.02, places=8)


if __name__ == "__main__":
    unittest.main()
