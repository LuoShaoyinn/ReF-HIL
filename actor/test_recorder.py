from __future__ import annotations

import unittest
from pathlib import Path
import pickle
import tempfile

import numpy as np
import torch

from .recorder import Recorder, RecorderConfig, build_classifier_preview, operator_has_intent, validate_demo_destination
from actor.success import ConditionResult, SuccessEvaluation


class _Task:
    def reset(self, _send_action: object, _read_observation: object) -> None:
        pass

    def prepare_observation(self, raw: dict) -> dict:
        return raw

    def build_operator_request(self, _raw: dict) -> dict:
        return {}

    def build_action_from_spacemouse(self, raw_action: dict, info: dict) -> torch.Tensor:
        del raw_action, info
        return torch.asarray([0.1, 0.0, 0.0, -1.0])

    def prepare_spacemouse_action(
        self, raw_action: dict, last_gripper_command: float
    ) -> dict:
        return {
            **raw_action,
            "gripper": float(last_gripper_command),
        }

    def parse_action(self, action: torch.Tensor) -> dict:
        return {
            "delta_pos": action[:3].numpy(),
            "delta_rot": np.zeros(3, dtype=np.float32),
            "gripper": float(action[3]),
        }

    def build_info(self, **kwargs: object) -> dict:
        return dict(kwargs)


class _Robot:
    def __init__(self) -> None:
        self.steps = 0

    def send_action(self, _action: dict) -> None:
        self.steps += 1

    def read_observation(self) -> dict:
        return {
            "tcp_speed": np.zeros(6, dtype=np.float32),
            "tcp_force": np.zeros(6, dtype=np.float32),
            "gripper": np.float32(-1.0),
            "projected_gravity": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            "images": {},
        }

    def close(self) -> None:
        pass


class _Operator:
    def reset(self) -> None:
        pass

    def read_action(self, _request: dict | None = None) -> dict:
        return {
            "delta_pos": np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
            "delta_rot": np.zeros(3, dtype=np.float32),
            "gripper_pressed": False,
        }

    def close(self) -> None:
        pass


class RecorderStartTest(unittest.TestCase):
    def test_append_saves_additional_success_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root/'0000.pkl'
            old.write_bytes(pickle.dumps([{'done':True,'existing':True}]))
            before = old.read_bytes()
            with self.assertRaises(FileExistsError):
                validate_demo_destination(root, append=False)
            robot = _Robot()
            recorder = Recorder(
                config=RecorderConfig(dataset_dir=root,append=True,max_episodes=1,
                                      max_steps_per_episode=1,loop_sleep_s=0.,operator_start_poll_s=0.),
                task=_Task(),robot=robot,operator=_Operator(),success_oracle=lambda _:True,
            )
            recorder.run()
            self.assertEqual(old.read_bytes(), before)
            with (root/'0001.pkl').open('rb') as stream:
                self.assertTrue(pickle.load(stream)[-1]['done'])
            self.assertEqual(robot.steps,1)

    def test_append_refuses_partial_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'0000.pkl').write_bytes(pickle.dumps([{'done':False}]))
            with self.assertRaisesRegex(ValueError,'episode boundary'):
                validate_demo_destination(root,append=True)

    def test_timeout_is_retried_without_consuming_success_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = _Robot()
            recorder = Recorder(
                config=RecorderConfig(
                    dataset_dir=root,
                    max_episodes=1,
                    max_steps_per_episode=1,
                    loop_sleep_s=0.0,
                    operator_start_poll_s=0.0,
                ),
                task=_Task(),
                robot=robot,
                operator=_Operator(),
                success_oracle=lambda _raw: robot.steps >= 2,
            )
            recorder.run()
            self.assertEqual(robot.steps, 2)
            saved_paths = sorted(root.glob("*.pkl"))
            self.assertEqual(len(saved_paths), 1)
            with saved_paths[0].open("rb") as stream:
                saved = pickle.load(stream)
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0]["done"])

    def test_failed_attempt_stops_at_horizon_and_is_not_committed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            robot = _Robot()
            recorder = Recorder(
                config=RecorderConfig(
                    dataset_dir=Path(directory),
                    max_steps_per_episode=1,
                    loop_sleep_s=0.0,
                    operator_start_poll_s=0.0,
                ),
                task=_Task(),
                robot=robot,
                operator=_Operator(),
                success_oracle=lambda _raw: robot.steps >= 3,
            )
            self.assertFalse(recorder.run_episode())
            self.assertEqual(robot.steps, 1)
            self.assertEqual(recorder.writer.pending, [])

    def test_successful_attempt_is_committed_as_one_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            robot = _Robot()
            recorder = Recorder(
                config=RecorderConfig(
                    dataset_dir=Path(directory),
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                    operator_start_poll_s=0.0,
                ),
                task=_Task(),
                robot=robot,
                operator=_Operator(),
                success_oracle=lambda _raw: robot.steps >= 3,
            )
            self.assertTrue(recorder.run_episode())
            self.assertEqual(len(recorder.writer.pending), 3)
            self.assertFalse(recorder.writer.pending[0]["done"])
            self.assertFalse(recorder.writer.pending[1]["done"])
            self.assertTrue(recorder.writer.pending[2]["done"])

    def test_classifier_preview_shows_rgb_and_depth(self) -> None:
        raw = {
            "images": {
                "classifier": np.zeros((8, 8, 3), dtype=np.uint8),
                "classifier_depth": np.full((8, 8), 127, dtype=np.uint8),
            }
        }
        preview = build_classifier_preview(
            raw,
            evaluation=SuccessEvaluation(
                conditions=(ConditionResult(
                    name="success",
                    score=0.8,
                    threshold=0.5,
                    passed=True,
                    image_key="classifier",
                    depth_key="classifier_depth",
                ),)
            ),
            scale=2,
        )
        self.assertEqual(preview.shape, (16, 32, 3))

    def test_classifier_preview_accepts_tcp_distance_condition(self):
        evaluation = SuccessEvaluation(conditions=(
            ConditionResult(name="image", score=.8, threshold=.5, passed=True,
                            image_key="classifier", depth_key="classifier_depth"),
            ConditionResult(name="tcp_position", score=.009, threshold=.010, passed=True,
                            image_key=None, depth_key=None),
        ))
        preview = build_classifier_preview(
            {"images": {"classifier": np.zeros((8, 8, 3), dtype=np.uint8)}},
            evaluation=evaluation, scale=2,
        )
        self.assertEqual(preview.shape, (64, 800, 3))

    def test_idle_gripper_state_does_not_start_recording(self) -> None:
        self.assertFalse(
            operator_has_intent(
                {
                    "delta_pos": np.zeros(3, dtype=np.float32),
                    "delta_rot": np.zeros(3, dtype=np.float32),
                    "gripper": -1.0,
                    "gripper_pressed": False,
                }
            )
        )

    def test_translation_starts_recording(self) -> None:
        self.assertTrue(
            operator_has_intent(
                {
                    "delta_pos": np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
                    "delta_rot": np.zeros(3, dtype=np.float32),
                    "gripper_pressed": False,
                }
            )
        )

    def test_rotation_or_gripper_press_starts_recording(self) -> None:
        self.assertTrue(
            operator_has_intent(
                {
                    "delta_pos": np.zeros(3, dtype=np.float32),
                    "delta_rot": np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
                    "gripper_pressed": False,
                }
            )
        )
        self.assertTrue(
            operator_has_intent(
                {
                    "delta_pos": np.zeros(3, dtype=np.float32),
                    "delta_rot": np.zeros(3, dtype=np.float32),
                    "gripper_pressed": True,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
