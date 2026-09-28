from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import select
import sys
import termios
import time
from typing import Callable, Protocol
import uuid

import tty

import torch

from actor.transport import RobotClient
from shared.protocol import DictMessage, validate_episode
from shared.task.base import BaseModeling
from shared.task.gripper_smoothness import tracker_for_task


class InferencePolicy(Protocol):
    def load(self, state: dict) -> None: ...

    def sample_action(self, observation: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class ValidationConfig:
    episodes_per_checkpoint: int = 20
    max_steps_per_episode: int
    loop_sleep_s: float = 0.1
    default_not_success_reward: float = -0.01


class TerminalSkipKey:
    """Non-blocking single-key reader for skipping physical validation attempts."""

    def __init__(self, key: str = "n", *, enabled: bool = True) -> None:
        if len(key) != 1:
            raise ValueError("skip key must be exactly one character")
        self.key = key
        self.enabled = bool(enabled)
        self._fd: int | None = None
        self._previous_settings: list | None = None

    def __enter__(self) -> "TerminalSkipKey":
        if not self.enabled:
            return self
        if not sys.stdin.isatty():
            print(
                "[VALIDATE] interactive skip disabled: stdin is not a TTY",
                flush=True,
            )
            return self
        self._fd = sys.stdin.fileno()
        self._previous_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        termios.tcflush(self._fd, termios.TCIFLUSH)
        print(
            f"[VALIDATE] press {self.key!r} at any time to mark the current "
            "attempt failed and reset (no Enter required)",
            flush=True,
        )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._fd is not None and self._previous_settings is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._previous_settings,
            )
        self._fd = None
        self._previous_settings = None

    def requested(self) -> bool:
        if self._fd is None:
            return False
        requested = False
        while select.select([self._fd], [], [], 0.0)[0]:
            character = sys.stdin.read(1)
            requested = requested or character.lower() == self.key.lower()
        return requested


def checkpoint_elapsed_seconds(path: Path) -> int:
    suffix = path.stem.removeprefix("checkpoint_").removesuffix("s")
    if not suffix.isdigit():
        raise ValueError(f"invalid checkpoint filename: {path.name}")
    return int(suffix)


def list_checkpoints(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("checkpoint_*s.pkl"),
        key=checkpoint_elapsed_seconds,
    )


def checkpoints_from(checkpoints: list[Path], start: str | None) -> list[Path]:
    """Select an exact checkpoint and every later elapsed-time checkpoint."""

    if start is None:
        return checkpoints
    token = Path(start).name
    if token.isdigit():
        elapsed_seconds = int(token)
    else:
        candidate = Path(token)
        try:
            elapsed_seconds = checkpoint_elapsed_seconds(candidate)
        except ValueError as error:
            raise ValueError(
                "--start-checkpoint must be elapsed seconds or a "
                "checkpoint_########s.pkl filename"
            ) from error
    matching = [
        path
        for path in checkpoints
        if checkpoint_elapsed_seconds(path) == elapsed_seconds
    ]
    if not matching:
        raise FileNotFoundError(
            f"start checkpoint at {elapsed_seconds}s was not found"
        )
    return [
        path
        for path in checkpoints
        if checkpoint_elapsed_seconds(path) >= elapsed_seconds
    ]


VALIDATION_SUMMARY_FORMAT = "checkpoint_validation_v1"
CHECKPOINT_PROFILE_FORMAT = "checkpoint_validation_profile_v1"
VALIDATION_EPISODE_FORMAT = "checkpoint_validation_episode_v1"


def latest_incomplete_run(directory: Path) -> str | None:
    """Return the newest resumable validation run id, if one exists."""

    candidates = sorted(
        Path(directory).glob("*_summary.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        resumable_format = payload.get("format") in (
            None,
            VALIDATION_SUMMARY_FORMAT,
        )
        if (
            resumable_format
            and not bool(payload.get("complete", False))
            and isinstance(payload.get("run_id"), str)
            and isinstance(payload.get("episodes_per_checkpoint"), int)
        ):
            return str(payload["run_id"])
    return None


class CheckpointValidator:
    """Run autonomous physical episodes without sending them to the learner."""

    def __init__(
        self,
        *,
        config: ValidationConfig,
        task: BaseModeling,
        policy: InferencePolicy,
        robot: RobotClient,
        success_oracle: Callable[[DictMessage], bool],
        result_directory: Path,
        skip_requested: Callable[[], bool] | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        if config.episodes_per_checkpoint < 1:
            raise ValueError("episodes_per_checkpoint must be positive")
        if config.max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be positive")
        self.config = config
        self.task = task
        self.policy = policy
        self.robot = robot
        self.success_oracle = success_oracle
        self.skip_requested = (
            (lambda: False) if skip_requested is None else skip_requested
        )
        self.result_directory = Path(result_directory)
        self.result_directory.mkdir(parents=True, exist_ok=True)
        if resume_run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        else:
            if not resume_run_id or Path(resume_run_id).name != resume_run_id:
                raise ValueError(f"invalid validation run id: {resume_run_id!r}")
            self.run_id = resume_run_id
        self.episode_log_path = self.result_directory / f"{self.run_id}.jsonl"
        self.summary_path = self.result_directory / f"{self.run_id}_summary.json"
        self._resuming = resume_run_id is not None
        self._checkpoint_files: list[str] = []

    def run(self, checkpoints: list[Path]) -> dict:
        if not checkpoints:
            raise FileNotFoundError("no checkpoint_########s.pkl files found")
        checkpoint_by_name = {path.name: path for path in checkpoints}
        if len(checkpoint_by_name) != len(checkpoints):
            raise ValueError("checkpoint filenames must be unique")
        if self._resuming:
            resume_state = self._load_resume_state(list(checkpoint_by_name))
            self._checkpoint_files = list(resume_state["checkpoint_files"])
            missing = [
                name for name in self._checkpoint_files if name not in checkpoint_by_name
            ]
            if missing:
                raise FileNotFoundError(
                    "cannot resume because planned checkpoints are missing: "
                    + ", ".join(missing)
                )
            checkpoints = [checkpoint_by_name[name] for name in self._checkpoint_files]
            results_by_checkpoint = self._load_existing_results()
            for checkpoint_name, existing_results in results_by_checkpoint.items():
                checkpoint_path = checkpoint_by_name[checkpoint_name]
                for result in existing_results:
                    self._update_checkpoint_profile(checkpoint_path, result)
            # This also upgrades a legacy pre-resume summary atomically before
            # the robot moves again. The JSONL remains the source of truth.
            self._write_summary(results_by_checkpoint, complete=False)
            print(
                f"[VALIDATE] resuming run {self.run_id} from "
                f"{sum(len(rows) for rows in results_by_checkpoint.values())} "
                "completed attempts",
                flush=True,
            )
        else:
            if self.summary_path.exists() or self.episode_log_path.exists():
                raise FileExistsError(f"validation run already exists: {self.run_id}")
            self._checkpoint_files = [path.name for path in checkpoints]
            results_by_checkpoint: dict[str, list[dict]] = {}
            self._write_summary(results_by_checkpoint, complete=False)
        try:
            for checkpoint_index, checkpoint_path in enumerate(checkpoints, 1):
                results = results_by_checkpoint.setdefault(checkpoint_path.name, [])
                if len(results) >= self.config.episodes_per_checkpoint:
                    print(
                        f"[VALIDATE] checkpoint {checkpoint_index}/{len(checkpoints)} "
                        f"already complete: {checkpoint_path.name}",
                        flush=True,
                    )
                    continue
                print(
                    f"[VALIDATE] checkpoint {checkpoint_index}/{len(checkpoints)}: "
                    f"{checkpoint_path.name}",
                    flush=True,
                )
                checkpoint = self._load_checkpoint(checkpoint_path)
                self.policy.load(checkpoint["actor"])
                for episode_index in range(
                    len(results) + 1,
                    self.config.episodes_per_checkpoint + 1,
                ):
                    started = time.monotonic()
                    episode, skipped = self.run_episode()
                    success = bool(
                        not skipped
                        and episode
                        and float(episode[-1]["reward"]) > 0.0
                    )
                    observed_return = float(
                        sum(row["reward"] for row in episode)
                    )
                    scored_steps = (
                        self.config.max_steps_per_episode
                        if skipped
                        else len(episode)
                    )
                    scored_return = (
                        float(self.config.default_not_success_reward)
                        * self.config.max_steps_per_episode
                        if skipped
                        else observed_return
                    )
                    result = {
                        "run_id": self.run_id,
                        "checkpoint": checkpoint_path.name,
                        "elapsed_seconds": int(checkpoint["elapsed_seconds"]),
                        "train_steps": int(checkpoint["learner"]["train_steps"]),
                        "episode": episode_index,
                        "success": success,
                        "skipped": bool(skipped),
                        "steps": len(episode),
                        "scored_steps": scored_steps,
                        "observed_return": observed_return,
                        "return": scored_return,
                        "wall_seconds": float(time.monotonic() - started),
                        "source_actor_episode": checkpoint.get("actor_episode"),
                        "source_policy_revision": checkpoint.get("policy_revision"),
                        "learner_actor_episodes": int(
                            checkpoint["learner"].get("actor_episodes", -1)
                        ),
                    }
                    result["outcome"] = (
                        "unsafe_failure"
                        if skipped
                        else "success"
                        if success
                        else "failure"
                    )
                    result["episode_artifact"] = self._write_episode_artifact(
                        checkpoint_path,
                        result,
                        episode,
                    )
                    self._append_result(result)
                    results.append(result)
                    self._update_checkpoint_profile(checkpoint_path, result)
                    self._write_summary(results_by_checkpoint, complete=False)
                    print(
                        f"[VALIDATE] episode {episode_index}/"
                        f"{self.config.episodes_per_checkpoint} "
                        f"success={result['success']} skipped={result['skipped']} "
                        f"steps={result['steps']}",
                        flush=True,
                    )
        finally:
            try:
                self.task.reset(
                    self.robot.send_action,
                    self.robot.read_observation,
                )
            finally:
                self.robot.close()
        final = self._write_summary(results_by_checkpoint, complete=True)
        print(f"[VALIDATE] results: {self.summary_path}", flush=True)
        return final

    def _load_resume_state(self, current_checkpoint_files: list[str]) -> dict:
        if not self.summary_path.exists():
            raise FileNotFoundError(
                f"resume summary does not exist: {self.summary_path}"
            )
        payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        if bool(payload.get("complete", False)):
            raise ValueError(f"validation run {self.run_id!r} is already complete")
        if int(payload.get("episodes_per_checkpoint", -1)) != int(
            self.config.episodes_per_checkpoint
        ):
            raise ValueError(
                "episodes_per_checkpoint differs from the interrupted run: "
                f"saved={payload.get('episodes_per_checkpoint')} "
                f"requested={self.config.episodes_per_checkpoint}"
            )
        summary_format = payload.get("format")
        if summary_format is None:
            # The old validator did not persist its checkpoint plan or maximum
            # episode length. Use the explicitly supplied/current checkpoint
            # directory, then rewrite the summary in the resumable format.
            payload["checkpoint_files"] = list(current_checkpoint_files)
            payload["max_steps_per_episode"] = int(
                self.config.max_steps_per_episode
            )
            print(
                f"[VALIDATE] upgrading legacy interrupted run {self.run_id}; "
                "using the checkpoints currently present in --checkpoint-dir",
                flush=True,
            )
        elif summary_format != VALIDATION_SUMMARY_FORMAT:
            raise ValueError(
                f"unsupported validation summary format: {summary_format!r}"
            )
        elif int(payload.get("max_steps_per_episode", -1)) != int(
            self.config.max_steps_per_episode
        ):
            raise ValueError(
                "max_steps_per_episode differs from the interrupted run: "
                f"saved={payload.get('max_steps_per_episode')} "
                f"requested={self.config.max_steps_per_episode}"
            )
        checkpoint_files = payload.get("checkpoint_files")
        if not isinstance(checkpoint_files, list) or not all(
            isinstance(name, str) for name in checkpoint_files
        ):
            raise ValueError("resume summary has no valid checkpoint plan")
        return payload

    def _load_existing_results(self) -> dict[str, list[dict]]:
        results: dict[str, list[dict]] = {}
        if not self.episode_log_path.exists():
            return results
        lines = self.episode_log_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    print(
                        "[VALIDATE] ignoring a truncated final JSONL record",
                        flush=True,
                    )
                    break
                raise
            if row.get("run_id") != self.run_id:
                raise ValueError("validation JSONL contains a different run id")
            checkpoint = row.get("checkpoint")
            if checkpoint not in self._checkpoint_files:
                raise ValueError(
                    f"validation JSONL contains unplanned checkpoint {checkpoint!r}"
                )
            results.setdefault(str(checkpoint), []).append(row)
        for checkpoint, rows in results.items():
            episode_indices = [int(row["episode"]) for row in rows]
            expected = list(range(1, len(rows) + 1))
            if episode_indices != expected:
                raise ValueError(
                    f"non-contiguous episodes for {checkpoint}: {episode_indices}"
                )
            if len(rows) > self.config.episodes_per_checkpoint:
                raise ValueError(f"too many episodes recorded for {checkpoint}")
        return results

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if not isinstance(checkpoint.get("actor"), dict):
            raise ValueError(f"checkpoint has no actor snapshot: {path}")
        return checkpoint

    def run_episode(self) -> tuple[list[DictMessage], bool]:
        self.task.reset(self.robot.send_action, self.robot.read_observation)
        smoothness = tracker_for_task(
            getattr(getattr(self.task, "config", None), "task", None)
        )
        episode: list[DictMessage] = []
        raw_obs = self.robot.read_observation()
        stored_obs = self.task.prepare_observation(raw_obs)
        observation = self.task.build_obs(stored_obs, {})
        for step_index in range(self.config.max_steps_per_episode):
            if self.skip_requested():
                print(
                    "[VALIDATE] operator marked current attempt as failure",
                    flush=True,
                )
                return episode, True
            previous_stored_obs = stored_obs
            action = self.policy.sample_action(observation)
            raw_action = self.task.parse_action(action)
            self.robot.send_action(raw_action)
            time.sleep(self.config.loop_sleep_s)

            raw_obs = self.robot.read_observation()
            smoothness.observe(float(raw_action["gripper"]))
            stored_obs = self.task.prepare_observation(raw_obs)
            observation = self.task.build_obs(stored_obs, {})
            success = bool(self.success_oracle(raw_obs))
            timeout = step_index + 1 >= self.config.max_steps_per_episode
            transition = {
                "raw_obs": previous_stored_obs,
                "raw_action": raw_action,
                "reward": smoothness.success_reward
                if success
                else float(self.config.default_not_success_reward),
                "done": bool(success or timeout),
                "info": self.task.build_info(
                    is_intervene=False,
                    steps_in_episode=step_index + 1,
                    max_steps_per_episode=self.config.max_steps_per_episode,
                    gripper_direction_reversals=smoothness.reversals,
                ),
            }
            if timeout and not success:
                transition["raw_next_obs"] = stored_obs
            episode.append(transition)
            if transition["done"]:
                break
        validate_episode(episode)
        return episode, False

    def _append_result(self, result: dict) -> None:
        with self.episode_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()

    def _profile_directory(self, checkpoint_path: Path) -> Path:
        return self.result_directory / "profiles" / checkpoint_path.stem

    def _write_episode_artifact(
        self,
        checkpoint_path: Path,
        result: dict,
        episode: list[DictMessage],
    ) -> str:
        """Persist one completed episode or explicit operator unsafe abort."""

        profile_directory = self._profile_directory(checkpoint_path)
        episode_directory = profile_directory / "episodes"
        episode_directory.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_id}_episode_{int(result['episode']):03d}.pkl"
        target = episode_directory / filename
        temporary = target.with_suffix(".pkl.tmp")
        payload = {
            "format": VALIDATION_EPISODE_FORMAT,
            "checkpoint": checkpoint_path.name,
            "run_id": self.run_id,
            "episode": int(result["episode"]),
            "outcome": result["outcome"],
            "result": dict(result),
            "transitions": episode,
        }
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        temporary.replace(target)
        return str(target.relative_to(self.result_directory))

    def _update_checkpoint_profile(
        self,
        checkpoint_path: Path,
        result: dict,
    ) -> None:
        result = dict(result)
        result.setdefault(
            "outcome",
            "unsafe_failure"
            if bool(result.get("skipped", False))
            else "success"
            if bool(result.get("success", False))
            else "failure",
        )
        profile_directory = self._profile_directory(checkpoint_path)
        profile_directory.mkdir(parents=True, exist_ok=True)
        profile_path = profile_directory / "profile.json"
        if profile_path.exists():
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            if payload.get("format") != CHECKPOINT_PROFILE_FORMAT:
                raise ValueError(f"unsupported checkpoint profile: {profile_path}")
            if payload.get("checkpoint") != checkpoint_path.name:
                raise ValueError(f"checkpoint profile mismatch: {profile_path}")
        else:
            payload = {
                "format": CHECKPOINT_PROFILE_FORMAT,
                "checkpoint": checkpoint_path.name,
                "elapsed_seconds": int(result["elapsed_seconds"]),
                "train_steps": int(result["train_steps"]),
                "attempts": [],
            }
        attempts = list(payload.get("attempts", []))
        attempt_id = f"{result['run_id']}:{int(result['episode'])}"
        profile_result = dict(result)
        profile_result["attempt_id"] = attempt_id
        attempts_by_id = {
            str(attempt["attempt_id"]): attempt
            for attempt in attempts
            if isinstance(attempt, dict) and "attempt_id" in attempt
        }
        attempts_by_id[attempt_id] = profile_result
        attempts = list(attempts_by_id.values())
        payload["attempts"] = attempts
        successes = sum(int(attempt["success"]) for attempt in attempts)
        payload["summary"] = {
            "attempts": len(attempts),
            "successes": successes,
            "failures": len(attempts) - successes,
            "unsafe_failures": sum(
                int(attempt.get("outcome") == "unsafe_failure")
                for attempt in attempts
            ),
            "success_rate": successes / len(attempts),
            "mean_scored_steps": sum(
                int(attempt["scored_steps"]) for attempt in attempts
            )
            / len(attempts),
            "mean_return": sum(float(attempt["return"]) for attempt in attempts)
            / len(attempts),
        }
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = profile_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(profile_path)

    @staticmethod
    def _summarize_checkpoint(checkpoint: str, results: list[dict]) -> dict:
        first = results[0]
        successes = sum(int(row["success"]) for row in results)
        return {
            "checkpoint": checkpoint,
            "elapsed_seconds": int(first["elapsed_seconds"]),
            "train_steps": int(first["train_steps"]),
            "source_actor_episode": first.get("source_actor_episode"),
            "source_policy_revision": first.get("source_policy_revision"),
            "learner_actor_episodes": int(
                first.get("learner_actor_episodes", -1)
            ),
            "episodes": len(results),
            "operator_failed_episodes": sum(
                int(row["skipped"]) for row in results
            ),
            "successes": successes,
            "success_rate": successes / len(results),
            "mean_steps": sum(
                row.get("scored_steps", row["steps"]) for row in results
            )
            / len(results),
            "mean_actual_steps": sum(row["steps"] for row in results)
            / len(results),
            "mean_return": sum(row["return"] for row in results) / len(results),
        }

    def _write_summary(
        self,
        results_by_checkpoint: dict[str, list[dict]],
        *,
        complete: bool,
    ) -> dict:
        summaries = [
            self._summarize_checkpoint(name, results_by_checkpoint[name])
            for name in self._checkpoint_files
            if results_by_checkpoint.get(name)
        ]
        payload = {
            "format": VALIDATION_SUMMARY_FORMAT,
            "run_id": self.run_id,
            "complete": bool(complete),
            "episodes_per_checkpoint": self.config.episodes_per_checkpoint,
            "max_steps_per_episode": self.config.max_steps_per_episode,
            "checkpoint_files": self._checkpoint_files,
            "checkpoints": summaries,
        }
        temporary = self.summary_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.summary_path)
        return payload
