from __future__ import annotations

import unittest

from .gripper_smoothness import GripperSmoothnessConfig, GripperSmoothnessTracker


class GripperSmoothnessTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = GripperSmoothnessTracker(GripperSmoothnessConfig())

    def test_direction_reversals_reduce_success_reward_to_floor(self) -> None:
        for command in (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0):
            self.tracker.observe(command)
        self.assertEqual(self.tracker.reversals, 6)
        self.assertAlmostEqual(self.tracker.success_reward, 0.5)

    def test_small_changes_accumulate_against_significant_anchor(self) -> None:
        self.assertFalse(self.tracker.observe(-0.95))
        self.assertFalse(self.tracker.observe(-0.91))
        self.assertFalse(self.tracker.observe(-0.89))
        self.assertEqual(self.tracker.direction, 1)
        self.assertEqual(self.tracker.reversals, 0)

if __name__ == "__main__":
    unittest.main()
