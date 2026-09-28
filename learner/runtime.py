from __future__ import annotations

import time
import torch
import pickle
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field

from learner.replay import DataBuffer, DataBufferConfig
from learner.subset_replay import SubsetDataBuffer
from shared.zmq import DictMessage, Receiver, Sender, ZmqEndpointConfig
from shared.task.base import BaseModeling
from shared.policy import BasePolicy
from learner.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointWriter,
    immutable_cpu_copy,
)


@dataclass(kw_only=True)
class LearnerConfig:
    transitions_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="0.0.0.0", port=7003)
    )
    actor_parameters_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="127.0.0.1", port=7004)
    )
    max_buffer_size:            int     = 200000
    offline_buffer_size:        int     = 200000
    batch_size:                 int     = 4096
    offline_batch_ratio:        float   = 0.5
    send_parameters_every:      int     = 2000
    save_every_seconds:         float   = 120.0
    checkpoint_async:           bool    = True
    log_every:                  int     = 2000
    experiment_name:            str     = "default"
    output_root:                Path    = Path("outputs")
    replay_device:              str     = "cuda"
    transitions_before_start:   int     = 200
    observation_noise_std:      float   = 0.03
    action_noise_std:           float   = 0.03


class Learner:
    def __init__(self, config: LearnerConfig, task: BaseModeling, policy: BasePolicy) -> None:
        self.config = config
        self.task = task
        self.policy = policy
        self.transition_receiver = Receiver(self.config.transitions_endpoint, timeout_ms=0)
        self.parameters_sender = Sender(self.config.actor_parameters_endpoint)

        self.output_dir = Path(self.config.output_root) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self.online_buffer = DataBuffer(
            DataBufferConfig(
                capacity=int(self.config.max_buffer_size),
                obs_dims=int(self.task.config.obs_dims),
                action_dims=int(self.task.config.action_dims),
                info_features={"is_intervene": (1,)},
                build_obs=self.task.build_obs,
                build_action=self.task.build_action,
                build_observations=getattr(self.task, "build_observations", None),
                build_actions=getattr(self.task, "build_actions", None),
                autosave_dir=self.output_dir / "buffer",
                data_device=str(self.config.replay_device),
                need_sample=True,
            )
        )
        self.offline_buffer = SubsetDataBuffer(self.online_buffer)
        for transition_idx, transition in self.online_buffer:
            if bool(transition["data_info"]["is_intervene"][0]):
                self.offline_buffer.add(transition_idx)

        self.train_steps = 0
        self.actor_steps = 0
        self.actor_episodes = 0
        self.training_elapsed_seconds = 0.0
        self._pending_checkpoint_extra_state: dict = {}
        self._train_metric_sums: dict[str, torch.Tensor] = {}
        self._train_metric_count = 0
        checkpoint_path = self._latest_checkpoint_path()
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        self._run_started_monotonic = time.monotonic()
        interval = float(self.config.save_every_seconds)
        self._next_save_monotonic = (
            self._run_started_monotonic + interval
            if interval > 0.0
            else float("inf")
        )
        self._checkpoint_writer = CheckpointWriter(
            asynchronous=bool(self.config.checkpoint_async and interval > 0.0)
        )

    def _latest_checkpoint_path(self) -> Path | None:
        latest_path: Path | None = None
        latest_seconds = -1
        for path in self.checkpoint_dir.glob("checkpoint_*s.pkl"):
            suffix = path.stem.removeprefix("checkpoint_").removesuffix("s")
            if not suffix.isdigit():
                continue
            elapsed_seconds = int(suffix)
            if elapsed_seconds > latest_seconds:
                latest_seconds = elapsed_seconds
                latest_path = path
        return latest_path

    def _load_checkpoint(self, path: Path) -> None:
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format: {path}")
        expected_class = type(self).__name__
        if state.get("class_name") != expected_class:
            raise ValueError(
                f"checkpoint class mismatch: {state.get('class_name')!r} "
                f"!= {expected_class!r}"
            )
        self.policy.load(state["policy"])
        learner_state = state["learner"]
        self.train_steps = int(learner_state["train_steps"])
        self.actor_steps = int(learner_state["actor_steps"])
        self.actor_episodes = int(learner_state["actor_episodes"])
        self.training_elapsed_seconds = float(state["elapsed_seconds"])
        replay_watermark = int(state["replay_watermark"])
        if self.online_buffer.raw_count < replay_watermark:
            raise RuntimeError(
                f"checkpoint requires {replay_watermark} replay rows, but only "
                f"{self.online_buffer.raw_count} were recovered"
            )
        self._pending_checkpoint_extra_state = dict(state.get("extra_state", {}))
        print(f"Loaded checkpoint {path}", flush=True)

    def _checkpoint_extra_state(self) -> dict:
        return {}

    def _elapsed_seconds(self) -> float:
        return self.training_elapsed_seconds + (
            time.monotonic() - self._run_started_monotonic
        )

    def _checkpoint_state_dict(self, elapsed_seconds: int) -> dict:
        return immutable_cpu_copy(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "class_name": type(self).__name__,
                "elapsed_seconds": int(elapsed_seconds),
                "learner": {
                    "train_steps": self.train_steps,
                    "actor_steps": self.actor_steps,
                    "actor_episodes": self.actor_episodes,
                },
                "policy": self.policy.export(),
                "actor": self._runtime_policy_export(),
                "extra_state": self._checkpoint_extra_state(),
                "replay_watermark": self.online_buffer.raw_count,
            }
        )

    def _flush_checkpoint_replay(self) -> None:
        self.online_buffer.save()

    def _save_checkpoint(self, *, wait: bool) -> bool:
        if not wait and not self._checkpoint_writer.available():
            return False
        self._flush_checkpoint_replay()
        elapsed_seconds = int(self._elapsed_seconds())
        path = self.checkpoint_dir / f"checkpoint_{elapsed_seconds:08d}s.pkl"
        state = self._checkpoint_state_dict(elapsed_seconds)
        submitted = self._checkpoint_writer.submit(path, state, wait=wait)
        if submitted:
            status = "saved" if wait else "queued"
            print(f"Checkpoint {status}: {path}", flush=True)
        return submitted

    def _save_model_and_state(self) -> None:
        """Compatibility name for explicit, complete synchronous saves."""

        self._save_checkpoint(wait=True)

    def _maybe_periodic_save(self) -> None:
        interval = float(self.config.save_every_seconds)
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now < self._next_save_monotonic:
            return
        while self._next_save_monotonic <= now:
            self._next_save_monotonic += interval
        self._save_checkpoint(wait=False)

    def _runtime_policy_export(self):
        export_actor = getattr(self.policy, "export_actor", None)
        snapshot = export_actor() if export_actor is not None else self.policy.export()
        return self._attach_runtime_metadata(snapshot)

    def _attach_runtime_metadata(self, snapshot: dict) -> dict:
        """Add algorithm-neutral provenance ignored by actor inference."""

        snapshot = dict(snapshot)
        snapshot["runtime"] = {
            "train_steps": int(self.train_steps),
            "actor_steps": int(self.actor_steps),
            "actor_episodes": int(self.actor_episodes),
            "elapsed_seconds": int(self._elapsed_seconds()),
        }
        return snapshot

    def _log_actor_transition(self, transition: DictMessage) -> None:
        self.actor_steps += 1
        if bool(transition["done"]):
            self.actor_episodes += 1
            self.writer.add_scalar("actor/reward", transition["reward"], self.actor_episodes)
        self.writer.add_scalar(
            "actor/is_intervene_actor",
            transition["info"]["is_intervene"],
            self.actor_steps,
        )

    def on_actor_transitions(self, transitions: list[DictMessage]) -> None:
        pass

    def _accumulate_train_metrics(self, result: DictMessage) -> None:
        for key, value in result.items():
            metric = value.detach().float().mean()
            if key in self._train_metric_sums:
                self._train_metric_sums[key].add_(metric)
            else:
                self._train_metric_sums[key] = metric.clone()
        self._train_metric_count += 1

    def _flush_train_metrics(self) -> None:
        if self._train_metric_count == 0:
            return
        for key, value_sum in self._train_metric_sums.items():
            self.writer.add_scalar(
                f"train/{key}",
                (value_sum / self._train_metric_count).item(),
                self.train_steps,
            )
        self._train_metric_sums.clear()
        self._train_metric_count = 0

    def step(self) -> None:
        self._maybe_periodic_save()
        transitions = self.transition_receiver.recv(None)
        if transitions is not None:
            self.on_actor_transitions(transitions)
            for transition in transitions:
                self._log_actor_transition(transition)
            transition_indices = self.online_buffer.add_many(transitions)
            for transition, transition_idx in zip(transitions, transition_indices, strict=True):
                if bool(transition["info"]["is_intervene"]):
                    self.offline_buffer.add(transition_idx)

        if self.online_buffer.tot_transition < max(2, int(self.config.transitions_before_start)):
            return

        batch_size = self.config.batch_size
        n_offline = min(
            max(1, int(batch_size * self.config.offline_batch_ratio)),
            max(0, self.offline_buffer.tot_transition - 1),
        )
        n_online = max(1, batch_size - n_offline)
        device = getattr(self.policy, "device", "cuda")

        online_batch = self.online_buffer.sample(n_online, device)
        if n_offline > 0:
            offline_batch = self.offline_buffer.sample(n_offline, device)
            batch: DictMessage = {"info": []}
            for key in ("observation", "next_observation", "action", "reward", "done"):
                batch[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
            batch["data_info"] = {
                key: torch.cat([online_batch["data_info"][key], offline_batch["data_info"][key]], dim=0)
                for key in online_batch["data_info"]
            }
        else:
            batch = online_batch

        batch["observation"] = batch["observation"] + (
            torch.randn_like(batch["observation"]) * self.config.observation_noise_std
        )
        batch["action"] = (
            batch["action"]
            + torch.randn_like(batch["action"]) * self.config.action_noise_std
        ).clamp_(-1.0, 1.0)

        result = self.policy.update(batch=batch)
        self._accumulate_train_metrics(result)
        self.train_steps += 1

        if self.train_steps % self.config.send_parameters_every == 0:
            self.parameters_sender.send(self._runtime_policy_export())
        if self.train_steps % self.config.log_every == 0:
            self._flush_train_metrics()
            print(f"[LEARNER] step={self.train_steps}")

    def run(self) -> None:
        try:
            print("Leaner started")
            while True:
                self.step()
        except KeyboardInterrupt as e:
            print("KeyboardInterrupted")
        finally:
            self._flush_train_metrics()
            self._save_model_and_state()
            self._checkpoint_writer.close()
            self.writer.flush()
            self.writer.close()
