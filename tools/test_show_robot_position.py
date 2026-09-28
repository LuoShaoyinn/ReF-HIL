from __future__ import annotations

import unittest

import numpy as np

from tools.show_robot_position import format_pose


class RobotPositionFormattingTest(unittest.TestCase):
    def test_format_includes_millimetres_and_rotation_vector(self) -> None:
        result = format_pose(np.asarray([-0.25, -0.65, 0.12, 2.0, 2.1, 0.1]))
        self.assertIn("xyz_mm=[-250.000, -650.000, 120.000]", result)
        self.assertIn("rotvec_rad=[2.000000, 2.100000, 0.100000]", result)


if __name__ == "__main__":
    unittest.main()
