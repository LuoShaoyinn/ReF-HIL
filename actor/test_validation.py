from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from actor.validation import (
    CheckpointValidator,
    ValidationConfig,
    checkpoints_from,
    latest_incomplete_run,
    list_checkpoints,
)
from actor.episode_snapshots import write_episode_actor
from actor.validate import next_validation_directory


class _Policy:
    def __init__(self) -> None:
        self.loaded: list[int] = []

    def load(self, state: dict) -> None:
        self.loaded.append(int(state["id"]))

    def sample_action(self, _observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(7)


class _Robot:
    def __init__(self) -> None:
        self.closed = False
        self.actions: list[dict] = []

    def read_observation(self) -> dict:
        return {"frame": len(self.actions)}

    def send_action(self, action: dict) -> None:
        self.actions.append(action)

    def close(self) -> None:
        self.closed = True


class _Task:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self, _send_action, _read_observation) -> None:
        self.reset_count += 1

    def prepare_observation(self, raw: dict) -> dict:
        return {
            "frame": raw["frame"],
            "tcp_speed": np.zeros(6, dtype=np.float32),
            "tcp_force": np.zeros(6, dtype=np.float32),
            "gripper": np.float32(-1.0),
            "projected_gravity": np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        }

    def build_obs(self, state: dict, _info: dict) -> torch.Tensor:
        return torch.tensor([float(state["frame"])])

    def parse_action(self, action: torch.Tensor) -> dict:
        values = action.detach().cpu().numpy()
        return {
            "delta_pos": values[:3],
            "delta_rot": values[3:6],
            "gripper": float(values[6]),
        }

    def build_info(self, **values) -> dict:
        return dict(values)


class ValidationTest(unittest.TestCase):
    def test_validation_directory_ids_never_reuse_interrupted_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-validation-id-") as tmp:
            base = Path(tmp) / "validate-task-all-0"
            artifact = Path("checkpoint_00000120s.pkl")
            first = next_validation_directory(base, artifact)
            second = next_validation_directory(base, artifact)
            self.assertEqual(first.name, "checkpoint_00000120s-000")
            self.assertEqual(second.name, "checkpoint_00000120s-001")

    def test_episode_actor_snapshot_validates_once_with_source_episode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-episode-validation-") as tmp:
            root = Path(tmp)
            snapshot = write_episode_actor(
                directory=root / "episode_actors",
                episode=7,
                policy_revision=3,
                actor_snapshot={
                    "id": 9,
                    "runtime": {
                        "train_steps": 123,
                        "actor_steps": 456,
                        "actor_episodes": 6,
                        "elapsed_seconds": 78,
                    },
                },
            )
            policy = _Policy()
            validator = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=1,
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                ),
                task=_Task(),
                policy=policy,
                robot=_Robot(),
                success_oracle=lambda _raw: True,
                result_directory=root / "validation",
            )
            summary = validator.run([snapshot])

            self.assertEqual(policy.loaded, [9])
            result = summary["checkpoints"][0]
            self.assertEqual(result["source_actor_episode"], 7)
            self.assertEqual(result["source_policy_revision"], 3)
            self.assertEqual(result["learner_actor_episodes"], 6)
            self.assertEqual(result["success_rate"], 1.0)

    def test_start_checkpoint_selects_named_checkpoint_and_later(self) -> None:
        checkpoints = [
            Path("checkpoint_00000120s.pkl"),
            Path("checkpoint_00000240s.pkl"),
            Path("checkpoint_00000360s.pkl"),
        ]
        expected = checkpoints[1:]
        self.assertEqual(checkpoints_from(checkpoints, "240"), expected)
        self.assertEqual(
            checkpoints_from(checkpoints, "checkpoint_00000240s.pkl"),
            expected,
        )

    def test_start_checkpoint_must_exist(self) -> None:
        with self.assertRaises(FileNotFoundError):
            checkpoints_from([Path("checkpoint_00000120s.pkl")], "240")

    def test_all_checkpoints_run_autonomously_and_log_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-validation-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            for seconds, identifier in ((240, 2), (120, 1)):
                with (
                    checkpoint_dir / f"checkpoint_{seconds:08d}s.pkl"
                ).open("wb") as handle:
                    pickle.dump(
                        {
                            "elapsed_seconds": seconds,
                            "learner": {"train_steps": seconds * 10},
                            "actor": {"id": identifier},
                        },
                        handle,
                    )
            policy = _Policy()
            robot = _Robot()
            task = _Task()
            validator = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=2,
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                ),
                task=task,
                policy=policy,
                robot=robot,
                success_oracle=lambda _raw: True,
                result_directory=root / "validation",
            )
            summary = validator.run(list_checkpoints(checkpoint_dir))

            self.assertEqual(policy.loaded, [1, 2])
            self.assertEqual(len(summary["checkpoints"]), 2)
            self.assertTrue(summary["complete"])
            self.assertTrue(robot.closed)
            self.assertEqual(task.reset_count, 5)
            lines = validator.episode_log_path.read_text().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertTrue(all(json.loads(line)["success"] for line in lines))
            self.assertTrue(all(not json.loads(line)["skipped"] for line in lines))
            for seconds in (120, 240):
                profile_directory = (
                    root
                    / "validation"
                    / "profiles"
                    / f"checkpoint_{seconds:08d}s"
                )
                profile = json.loads(
                    (profile_directory / "profile.json").read_text()
                )
                self.assertEqual(profile["summary"]["attempts"], 2)
                self.assertEqual(profile["summary"]["success_rate"], 1.0)
                artifacts = sorted((profile_directory / "episodes").glob("*.pkl"))
                self.assertEqual(len(artifacts), 2)
                with artifacts[0].open("rb") as handle:
                    artifact = pickle.load(handle)
                self.assertEqual(artifact["outcome"], "success")
                self.assertEqual(len(artifact["transitions"]), 1)
            loaded_summary = json.loads(validator.summary_path.read_text())
            self.assertEqual(
                [row["elapsed_seconds"] for row in loaded_summary["checkpoints"]],
                [120, 240],
            )

    def test_operator_skip_counts_as_a_timeout_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-validation-skip-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint_path = checkpoint_dir / "checkpoint_00000120s.pkl"
            with checkpoint_path.open("wb") as handle:
                pickle.dump(
                    {
                        "elapsed_seconds": 120,
                        "learner": {"train_steps": 1200},
                        "actor": {"id": 1},
                    },
                    handle,
                )
            requests = iter((True, False))
            validator = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=2,
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                ),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                success_oracle=lambda _raw: True,
                result_directory=root / "validation",
                skip_requested=lambda: next(requests, False),
            )

            summary = validator.run([checkpoint_path])

            checkpoint = summary["checkpoints"][0]
            self.assertEqual(checkpoint["episodes"], 2)
            self.assertEqual(checkpoint["operator_failed_episodes"], 1)
            self.assertEqual(checkpoint["successes"], 1)
            self.assertEqual(checkpoint["success_rate"], 0.5)
            self.assertEqual(checkpoint["mean_steps"], 2.0)
            self.assertEqual(checkpoint["mean_actual_steps"], 0.5)
            lines = [
                json.loads(line)
                for line in validator.episode_log_path.read_text().splitlines()
            ]
            self.assertTrue(lines[0]["skipped"])
            self.assertFalse(lines[0]["success"])
            self.assertEqual(lines[0]["outcome"], "unsafe_failure")
            self.assertEqual(lines[0]["scored_steps"], 3)
            self.assertAlmostEqual(lines[0]["return"], -0.03)
            self.assertFalse(lines[1]["skipped"])
            profile_path = (
                root
                / "validation"
                / "profiles"
                / "checkpoint_00000120s"
                / "profile.json"
            )
            profile = json.loads(profile_path.read_text())
            self.assertEqual(profile["summary"]["attempts"], 2)
            self.assertEqual(profile["summary"]["failures"], 1)
            self.assertEqual(profile["summary"]["unsafe_failures"], 1)

    def test_interrupted_run_resumes_at_next_unrecorded_episode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-validation-resume-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint_path = checkpoint_dir / "checkpoint_00000120s.pkl"
            with checkpoint_path.open("wb") as handle:
                pickle.dump(
                    {
                        "elapsed_seconds": 120,
                        "learner": {"train_steps": 1200},
                        "actor": {"id": 1},
                    },
                    handle,
                )

            checks = 0

            def interrupt_second_episode() -> bool:
                nonlocal checks
                checks += 1
                if checks == 2:
                    raise KeyboardInterrupt
                return False

            first = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=2,
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                ),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                success_oracle=lambda _raw: True,
                result_directory=root / "validation",
                skip_requested=interrupt_second_episode,
            )
            with self.assertRaises(KeyboardInterrupt):
                first.run([checkpoint_path])

            interrupted = json.loads(first.summary_path.read_text())
            self.assertFalse(interrupted["complete"])
            self.assertEqual(interrupted["checkpoints"][0]["episodes"], 1)
            self.assertEqual(
                latest_incomplete_run(root / "validation"), first.run_id
            )

            resumed_policy = _Policy()
            resumed = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=2,
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                ),
                task=_Task(),
                policy=resumed_policy,
                robot=_Robot(),
                success_oracle=lambda _raw: True,
                result_directory=root / "validation",
                resume_run_id=first.run_id,
            )
            summary = resumed.run([checkpoint_path])

            self.assertTrue(summary["complete"])
            self.assertEqual(summary["checkpoints"][0]["episodes"], 2)
            self.assertEqual(summary["checkpoints"][0]["successes"], 2)
            self.assertEqual(resumed_policy.loaded, [1])
            lines = resumed.episode_log_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                [json.loads(line)["episode"] for line in lines], [1, 2]
            )
            self.assertIsNone(latest_incomplete_run(root / "validation"))

    def test_legacy_interrupted_run_is_rebuilt_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-validation-legacy-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint_path = checkpoint_dir / "checkpoint_00000120s.pkl"
            with checkpoint_path.open("wb") as handle:
                pickle.dump(
                    {
                        "elapsed_seconds": 120,
                        "learner": {"train_steps": 1200},
                        "actor": {"id": 1},
                    },
                    handle,
                )
            result_dir = root / "validation"
            result_dir.mkdir()
            run_id = "20260822T000000Z-legacy01"
            legacy_row = {
                "run_id": run_id,
                "checkpoint": checkpoint_path.name,
                "elapsed_seconds": 120,
                "train_steps": 1200,
                "episode": 1,
                "success": True,
                "skipped": False,
                "steps": 1,
                "scored_steps": 1,
                "observed_return": 1.0,
                "return": 1.0,
                "wall_seconds": 0.1,
            }
            (result_dir / f"{run_id}.jsonl").write_text(
                json.dumps(legacy_row) + "\n",
                encoding="utf-8",
            )
            (result_dir / f"{run_id}_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "complete": False,
                        "episodes_per_checkpoint": 2,
                        "checkpoints": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(latest_incomplete_run(result_dir), run_id)

            resumed = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=2,
                    max_steps_per_episode=3,
                    loop_sleep_s=0.0,
                ),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                success_oracle=lambda _raw: True,
                result_directory=result_dir,
                resume_run_id=run_id,
            )
            summary = resumed.run([checkpoint_path])

            self.assertTrue(summary["complete"])
            self.assertEqual(summary["format"], "checkpoint_validation_v1")
            self.assertEqual(summary["checkpoint_files"], [checkpoint_path.name])
            self.assertEqual(summary["checkpoints"][0]["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
    checkpoints_from,
