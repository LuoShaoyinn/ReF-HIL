from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
import time
import threading
from typing import Callable

from shared.zmq import DictMessage, Receiver, Sender, ZmqEndpointConfig
from shared.task.base import BaseModeling
from shared.task.gripper_smoothness import tracker_for_task
from shared.policy import BasePolicy
from actor.transport import OperatorClient, RobotClient
from actor.episode_snapshots import next_episode_actor_index, write_episode_actor
from shared.protocol import validate_episode


@dataclass(kw_only=True)
class ActorConfig:
    learner_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="192.168.1.100", port=7003)
    )
    parameters_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="0.0.0.0", port=7004)
    )
    max_episodes: int = 1000000
    max_steps_per_episode: int
    loop_sleep_s: float = 0.1
    parameters_poll_timeout_ms: int = 100
    wait_for_initial_parameters: bool = True
    default_not_success_reward: float = -0.01
    episode_actor_directory: Path | None = None


class Actor(ABC):
    def __init__(
        self,
        config: ActorConfig,
        task: BaseModeling,
        policy: BasePolicy,
        robot: RobotClient,
        operator: OperatorClient,
        success_oracle: Callable[[DictMessage], bool],
        force_success_requested: Callable[[], bool] | None = None,
        drain_force_success: Callable[[], None] | None = None,
        force_failure_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.task = task
        self.policy = policy
        self.robot = robot
        self.operator = operator
        self.success_oracle = success_oracle
        self.drain_force_success = drain_force_success or (lambda: None)
        self.force_failure_requested = force_failure_requested or (lambda: False)
        self.force_success_requested = (
            (lambda: False)
            if force_success_requested is None
            else force_success_requested
        )
        self.learner_sender = Sender(self.config.learner_endpoint)
        self.parameters_receiver = Receiver(
            self.config.parameters_endpoint,
            timeout_ms=self.config.parameters_poll_timeout_ms,
        )
        self.last_gripper_command = -1.0
        self._current_actor_snapshot: dict | None = None
        self._policy_revision = 0

    def run(self) -> None:
        try:
            initial_policy_update = self._sync_parameters(
                block=bool(self.config.wait_for_initial_parameters)
            )
            first_episode = (
                1
                if self.config.episode_actor_directory is None
                else next_episode_actor_index(self.config.episode_actor_directory)
            )
            for local_episode_idx in range(self.config.max_episodes):
                episode_idx = first_episode + local_episode_idx
                # Preserve the original pre-episode poll, so a snapshot that
                # arrived just before control starts is never delayed.
                policy_updated = self._sync_parameters(block=False)
                if local_episode_idx == 0:
                    policy_updated = policy_updated or initial_policy_update
                self._save_episode_actor(episode_idx)
                progress_stop, progress_thread = self._start_episode_progress(
                    episode_idx
                )
                episode: list[DictMessage] | None = None
                try:
                    episode = self.run_episode()
                    # Apply the newest learner snapshot between episodes, so
                    # the next physical rollout uses it.  The terminal line
                    # reports that update where the operator is already
                    # looking for the episode outcome.
                    policy_updated = (
                        self._sync_parameters(block=False) or policy_updated
                    )
                finally:
                    if episode is None:
                        self._finish_episode_progress(
                            progress_stop,
                            progress_thread,
                            outcome="interrupted",
                            policy_updated=False,
                        )
                    else:
                        self._finish_episode_progress(
                            progress_stop,
                            progress_thread,
                            outcome=(
                                "success=True"
                                if episode and float(episode[-1]["reward"]) > 0.0
                                else "success=False"
                            ),
                            policy_updated=policy_updated,
                        )
            # Don't need to save
        except KeyboardInterrupt:
            print("Actor interrupted by user. Exiting...")
        finally:
            self.task.reset(self.robot.send_action, self.robot.read_observation)
            self.robot.close()
            self.operator.close()
            self.learner_sender.close()
            self.parameters_receiver.close()

    def run_episode(self) -> list[DictMessage]:
        self.task.reset(self.robot.send_action, self.robot.read_observation)
        self.operator.reset()
        self.last_gripper_command = -1.0
        smoothness = tracker_for_task(
            getattr(getattr(self.task, "config", None), "task", None),
            initial_command=self.last_gripper_command,
        )
        episode: list[DictMessage] = []
        raw_obs = self.robot.read_observation()
        stored_obs = self.task.prepare_observation(raw_obs)
        obs = self.task.build_obs(stored_obs, {})
        self.drain_force_success()
        for step_idx in range(self.config.max_steps_per_episode):
            if self.force_failure_requested():
                if episode:
                    episode[-1].update(reward=float(self.config.default_not_success_reward), done=True, raw_next_obs=stored_obs)
                    episode[-1]["info"]["forced_failure"] = True
                    episode[-1]["info"]["unrecoverable_next"] = True
                    episode[-1]["info"].pop("forced_success", None)
                else:
                    # No fictitious action/transition: send a separate state annotation.
                    self.learner_sender.send(
                        {"event": "failed_state", "raw_obs": stored_obs, "unrecoverable": True}
                    )
                print("[ACTOR] f: current state unrecoverable; episode terminated", flush=True)
                break
            forced_before_action = bool(self.force_success_requested())
            if forced_before_action and episode:
                episode[-1]["reward"] = smoothness.success_reward
                episode[-1]["done"] = True
                episode[-1]["info"]["forced_success"] = True
                print("[ACTOR] episode forced successful by Enter", flush=True)
                break
            prev_stored_obs = stored_obs
            policy_action = self.policy.sample_action(obs)
            human_action = self.operator.read_action(
                self.task.build_operator_request(raw_obs)
            )
            action, is_intervene = self.task.select_action(
                policy_action,
                human_action,
                self.last_gripper_command,
            )
            raw_action = self.task.parse_action(action)
            self.robot.send_action(raw_action)
            time.sleep(self.config.loop_sleep_s)

            raw_obs = self.robot.read_observation()
            smoothness.observe(float(raw_action["gripper"]))
            stored_obs = self.task.prepare_observation(raw_obs)
            obs = self.task.build_obs(stored_obs, {})
            forced_success = bool(
                forced_before_action or self.force_success_requested()
            )
            forced_failure = bool(self.force_failure_requested())
            success = bool(
                not forced_failure and (self.success_oracle(raw_obs) or forced_success)
            )
            reward = (
                smoothness.success_reward
                if success
                else float(self.config.default_not_success_reward)
            )
            timeout = (step_idx + 1) >= int(self.config.max_steps_per_episode)
            done = bool(success or timeout or forced_failure)
            if forced_failure:
                reward = float(self.config.default_not_success_reward)
            info = self.task.build_info(
                is_intervene=is_intervene,
                steps_in_episode=step_idx + 1,
                max_steps_per_episode=int(self.config.max_steps_per_episode),
                gripper_direction_reversals=smoothness.reversals,
            )
            if forced_failure:
                info["forced_failure"] = True
                info["unrecoverable_next"] = True
                print("[ACTOR] f: current state unrecoverable; episode terminated", flush=True)
            elif forced_success:
                info["forced_success"] = True
                print("[ACTOR] episode forced successful by Enter", flush=True)
            if is_intervene:
                # Preserve the counterfactual proposal at exactly the same
                # observation as the executed human correction.  The learner
                # uses only autonomous->intervention boundaries for ranking.
                info["autonomous_action"] = self.task.parse_action(policy_action)
            transition = {
                "raw_obs": prev_stored_obs,
                "raw_action": raw_action,  # already a dict
                "reward": reward,
                "done": done,
                "info": info,
            }
            # Success is a true terminal. Timeout is only a collection
            # boundary, so preserve the physical post-action state for the
            # vector Bellman shift at horizons greater than one.
            if (timeout and not success) or forced_failure:
                transition["raw_next_obs"] = stored_obs
            episode.append(transition)
            self.last_gripper_command = float(raw_action["gripper"])
            if done:
                break

        if episode:
            validate_episode(episode)
            self.learner_sender.send(episode)
        return episode

    @staticmethod
    def _start_episode_progress(
        episode_index: int,
    ) -> tuple[threading.Event, threading.Thread]:
        """Render one dot every 0.5 s while a physical episode is active."""

        stop = threading.Event()
        print(f"[{episode_index}] ", end="", flush=True)

        def render_dots() -> None:
            while not stop.wait(0.5):
                print(".", end="", flush=True)

        thread = threading.Thread(
            target=render_dots,
            name=f"actor-episode-{episode_index}-progress",
            daemon=True,
        )
        thread.start()
        return stop, thread

    @staticmethod
    def _finish_episode_progress(
        stop: threading.Event,
        thread: threading.Thread,
        *,
        outcome: str,
        policy_updated: bool,
    ) -> None:
        stop.set()
        thread.join()
        update = " policy=updated" if policy_updated else ""
        print(f"  {outcome}{update}", flush=True)

    def _sync_parameters(self, block: bool) -> bool:
        self.parameters_receiver.timeout_ms = -1 if block else 0
        last_params = self.parameters_receiver.recv(None)
        while True and not block:
            params = self.parameters_receiver.recv(None)
            if params is None:
                break
            last_params = params
        if last_params is not None:
            self.policy.load(last_params)
            self._current_actor_snapshot = last_params
            self._policy_revision += 1
            return True
        return False

    def _save_episode_actor(self, episode: int) -> None:
        directory = self.config.episode_actor_directory
        if directory is None:
            return
        if self._current_actor_snapshot is None:
            raise RuntimeError("cannot save an episode actor before parameters load")
        path = write_episode_actor(
            directory=directory,
            episode=episode,
            policy_revision=self._policy_revision,
            actor_snapshot=self._current_actor_snapshot,
        )
        print(f"[ACTOR] episode policy: {path}", flush=True)
