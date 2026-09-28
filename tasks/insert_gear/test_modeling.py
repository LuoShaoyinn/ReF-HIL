from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from .modeling import Modeling, ModelingConfig, rotation_at_yaw


class InsertGearModelingTest(unittest.TestCase):
    def setUp(self):
        self.modeling = object.__new__(Modeling)
        self.modeling.config = ModelingConfig(device="cpu")
        self.modeling.device = torch.device("cpu")

    def test_action_excludes_fixed_gripper(self):
        action = self.modeling.build_action({"delta_pos": [0.1, -0.2, 0.3],
                                            "delta_rot": [0.8, -0.7, 0.4], "gripper": -1}, {})
        np.testing.assert_allclose(action.numpy(), [0.1, -0.2, 0.3, 0.4])
        self.assertEqual(self.modeling.parse_action(action)["gripper"], 1.0)

    def test_release_regrasp_sequence_above_below_and_at_threshold(self):
        task = self.modeling.config.task
        for z in (0.060, 0.040, task.reset_release_z):
            with self.subTest(z=z):
                initial = np.asarray([[-0.12, -0.47, z], rotation_at_yaw(task, 0.5)], dtype=np.float32)
                pose = initial.copy()
                events = []

                def send(action):
                    events.append(action)
                    if "reset_target_pose" in action:
                        pose[:] = action["reset_target_pose"]

                with patch("tasks.insert_gear.modeling.time.sleep", side_effect=lambda t: events.append(t)):
                    self.modeling.reset(send, lambda: {"tcp_pose": pose.copy()})
                actions = [e for e in events if isinstance(e, dict)]
                opens = [i for i, e in enumerate(actions) if e.get("gripper_only") and e["gripper"] == -1]
                closes = [i for i, e in enumerate(actions) if e.get("gripper_only") and e["gripper"] == 1]
                self.assertEqual(len(opens), 1)
                self.assertEqual(len(closes), 1)
                for value, wait in ((-1.0, 1.0), (1.0, 2.0)):
                    event_index = next(i for i, e in enumerate(events)
                                       if isinstance(e, dict) and e.get("gripper_only") and e["gripper"] == value)
                    self.assertEqual(events[event_index + 1], wait)
                    self.assertIn("reset_target_pose", events[event_index + 2])
                self.assertEqual(opens[0], 1 if z > task.reset_release_z else 0)
                if z > task.reset_release_z:
                    np.testing.assert_allclose(actions[0]["reset_target_pose"][0], [*initial[0, :2], task.reset_release_z])
                    np.testing.assert_allclose(actions[0]["reset_target_pose"][1], initial[1])
                between = actions[opens[0] + 1:closes[0]]
                self.assertTrue(all(e["gripper"] == -1 for e in between))
                first_lift = between[0]["reset_target_pose"]
                expected_xy = initial[0, :2]
                np.testing.assert_allclose(first_lift[0], [*expected_xy, task.reset_pos[2]])
                np.testing.assert_allclose(first_lift[1], initial[1])
                np.testing.assert_allclose(between[1]["reset_target_pose"][1], rotation_at_yaw(task, 0))
                wait_index = next(i for i, e in enumerate(events) if isinstance(e, float) and e == 5.0)
                before_wait = [e for e in events[:wait_index] if isinstance(e, dict)]
                np.testing.assert_allclose(before_wait[-1]["reset_target_pose"][0], first_lift[0])
                np.testing.assert_allclose(between[-2]["reset_target_pose"][0], [*task.startup_pickup_pos[:2], task.reset_pos[2]])
                np.testing.assert_allclose(between[-1]["reset_target_pose"][0], task.startup_pickup_pos)
                after_close = actions[closes[0] + 1:]
                self.assertTrue(all(e["gripper"] == 1 for e in after_close))
                np.testing.assert_allclose(after_close[0]["reset_target_pose"][0], [*task.startup_pickup_pos[:2], task.reset_pos[2]])
                self.assertTrue(np.all(pose[0] >= np.asarray(task.safety_pos_min, dtype=np.float32)))
                self.assertTrue(np.all(pose[0] <= np.asarray(task.safety_pos_max, dtype=np.float32)))
                self.assertEqual(events[-1], task.reset_settle_time_s)

    def test_failed_stage_stops_remaining_sequence(self):
        task = self.modeling.config.task
        initial = np.asarray([[-0.12, -0.47, 0.060], task.zero_point_rot], dtype=np.float32)
        stages = ("release descent", "open-gripper lift", "rotation alignment", "pickup approach",
                  "pickup descent", "closed-gripper lift", "randomized start")
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                sent = []
                with patch.object(self.modeling, "_move_reset_pose", side_effect=[True] * index + [False]) as move, patch("tasks.insert_gear.modeling.time.sleep"):
                    with self.assertRaisesRegex(RuntimeError, stage):
                        self.modeling.reset(sent.append, lambda: {"tcp_pose": initial.copy()})
                self.assertEqual(move.call_count, index + 1)
                if index <= 4:
                    self.assertFalse(any(a["gripper"] == 1 for a in sent))


if __name__ == "__main__":
    unittest.main()
