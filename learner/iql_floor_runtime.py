from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from learner.replay import DataBuffer, DataBufferConfig
from learner.runtime import Learner, LearnerConfig
from learner.subset_replay import RawIndexBuffer, RawIndexDataView
from shared.zmq import DictMessage

from .iql_modules import (
    build_vector_bellman_target,
    horizon_extension_violation,
)
from .iql_floor import HumanIQLGroundedPolicy


@dataclass(kw_only=True)
class HumanIQLGroundedLearnerConfig(LearnerConfig):
    # Preserve the previous human-value-floor runtime defaults.
    send_parameters_every: int = 100
    human_example_capacity: int = 20_000
    human_example_batch_size: int = 512
    human_example_min_steps: int = 1
    human_iql_episode_count: int = 20
    human_iql_updates: int = 20_000
    human_iql_log_every: int = 250
    q_actor_warmup_updates: int = 5_000
    replay_exposure_power: float = 1.0
    replay_uniform_mix: float = 0.10
    indexed_replay_views: bool = False


class HumanIQLGroundedLearner(Learner):
    """Learner with moving human-IQL grounding and Q-only actor learning."""

    config: HumanIQLGroundedLearnerConfig
    policy: HumanIQLGroundedPolicy

    def __init__(self, config, task, policy) -> None:
        super().__init__(config=config, task=task, policy=policy)
        if self.policy.loaded_legacy_iql:
            raise RuntimeError(
                "cannot resume training from a legacy shared SAC/IQL "
                "checkpoint; start a new experiment directory from copied "
                "raw replay so independent Q_IQL can be trained from scratch"
            )
        if not 0.0 <= config.replay_uniform_mix <= 1.0:
            raise ValueError("replay_uniform_mix must be in [0,1]")
        if config.replay_exposure_power < 0.0:
            raise ValueError("replay_exposure_power must be non-negative")

        replay_capacity = self.online_buffer.config.capacity
        self.replay_sample_counts = torch.zeros(
            replay_capacity,
            dtype=torch.float32,
            device=self.online_buffer.config.data_device,
        )
        self.timeout_next_id = torch.full(
            (replay_capacity,),
            -1,
            dtype=torch.long,
            device=self.online_buffer.config.data_device,
        )
        self.timeout_next_observations = torch.empty(
            (0, int(task.config.obs_dims)),
            dtype=torch.float32,
            device=self.online_buffer.config.data_device,
        )
        self.timeout_next_count = 0
        self.has_timeout_next_rows = False

        if config.indexed_replay_views:
            self.offline_buffer = RawIndexBuffer(
                self.online_buffer, name="D_human"
            )
            self.human_example_buffer = RawIndexDataView(
                self.online_buffer, name="D_human_suffix"
            )
            self._rebuild_raw_index_views()
        else:
            self.human_example_buffer = DataBuffer(
                DataBufferConfig(
                    capacity=int(config.human_example_capacity),
                    obs_dims=int(task.config.obs_dims),
                    action_dims=int(task.config.action_dims),
                    info_features={
                        "is_intervene": (1,),
                    },
                    build_obs=task.build_obs,
                    build_action=task.build_action,
                    build_observations=getattr(task, "build_observations", None),
                    build_actions=getattr(task, "build_actions", None),
                    autosave_dir=self.output_dir / "human_example",
                    data_device=str(config.replay_device),
                    need_sample=True,
                )
            )
        self.human_sample_counts = torch.zeros(
            config.human_example_capacity,
            dtype=torch.float32,
            device=self.human_example_buffer.config.data_device,
        )
        self.human_example_episodes = self._human_episode_count()
        if self.human_example_buffer.tot_transition == 0:
            self._seed_human_examples_from_replay()
        self.first20_cutoff = self._human_iql_cutoff()
        self._refresh_human_candidates()
        self._refresh_replay_candidates()
        self._load_algorithm_state()
        self._rebuild_timeout_next_observations()
        if not self.policy.human_iql_initialized:
            self._pretrain_human_iql()

    def _rebuild_raw_index_views(self) -> None:
        rows = self._load_raw_replay(self.output_dir / "buffer")
        episode_rows: list[DictMessage] = []
        episode_ids: list[int] = []
        for raw_id, row in enumerate(rows):
            if bool(row.get("info", {}).get("is_intervene", False)):
                self.offline_buffer.add(raw_id)
            episode_rows.append(row)
            episode_ids.append(raw_id)
            if not bool(row.get("done", False)):
                continue
            suffix = self._complete_human_suffix(
                episode_rows, self.config.human_example_min_steps
            )
            if suffix:
                start = len(episode_rows) - len(suffix)
                self.human_example_buffer.add_many(episode_ids[start:])
            episode_rows = []
            episode_ids = []

    def _load_algorithm_state(self) -> None:
        payload = self._pending_checkpoint_extra_state
        if not payload:
            return
        for name, target in (
            ("replay_sample_counts", self.replay_sample_counts),
            ("human_sample_counts", self.human_sample_counts),
        ):
            source = payload.get(name)
            if source is None:
                continue
            source = torch.as_tensor(source, dtype=target.dtype)
            count = min(len(source), len(target))
            target[:count].copy_(source[:count].to(target.device))
        self.first20_cutoff = int(payload.get("first20_cutoff", self.first20_cutoff))
        self._pending_checkpoint_extra_state = {}

    def _checkpoint_extra_state(self) -> dict:
        if not hasattr(self, "replay_sample_counts"):
            return {}
        return {
            "first20_cutoff": self.first20_cutoff,
            "replay_sample_counts": self.replay_sample_counts[
                : self.online_buffer.tot_transition
            ],
            "human_sample_counts": self.human_sample_counts[
                : self.human_example_buffer.tot_transition
            ],
        }

    def _flush_checkpoint_replay(self) -> None:
        super()._flush_checkpoint_replay()
        if hasattr(self, "human_example_buffer"):
            self.human_example_buffer.save()

    def _runtime_policy_export(self):
        return self._attach_runtime_metadata(self.policy.export_actor())

    @staticmethod
    def _load_raw_replay(buffer_dir: Path) -> list[DictMessage]:
        rows: list[DictMessage] = []
        for path in sorted(buffer_dir.glob("*.pkl")):
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, list):
                raise TypeError(f"Replay chunk is not a list: {path}")
            rows.extend(cast(list[DictMessage], payload))
        return rows

    @staticmethod
    def _complete_human_suffix(
        transitions: list[DictMessage],
        minimum_steps: int,
    ) -> list[DictMessage]:
        if not transitions:
            return []
        final = transitions[-1]
        if not bool(final.get("done", False)):
            return []
        if final.get("info", {}).get("unrecoverable_next", False):
            if float(final.get("reward", 0.0)) > 0:
                raise ValueError("unrecoverable episode cannot end in success")
            return [copy.deepcopy(transition) for transition in transitions]
        if float(final.get("reward", 0.0)) <= 0.0:
            return []
        start = len(transitions)
        while start > 0 and bool(
            transitions[start - 1].get("info", {}).get("is_intervene", False)
        ):
            start -= 1
        suffix = transitions[start:]
        if len(suffix) < minimum_steps:
            return []
        return [copy.deepcopy(transition) for transition in suffix]

    def _seed_human_examples_from_replay(self) -> None:
        episode: list[DictMessage] = []
        episodes = 0
        transitions = 0
        for row in self._load_raw_replay(self.output_dir / "buffer"):
            episode.append(row)
            if not bool(row.get("done", False)):
                continue
            suffix = self._complete_human_suffix(
                episode, self.config.human_example_min_steps
            )
            if suffix:
                self.human_example_buffer.add_many(suffix)
                episodes += 1
                transitions += len(suffix)
            episode = []
        self.human_example_buffer.save()
        self.human_example_episodes = episodes
        if transitions:
            print(
                "[HUMAN IQL] seeded "
                f"{episodes} success suffixes / {transitions} transitions",
                flush=True,
            )

    def _human_episode_count(self) -> int:
        done = self.human_example_buffer.data_done
        if done is None:
            return 0
        return int(
            done[: self.human_example_buffer.tot_transition]
            .sum()
            .detach()
            .cpu()
            .item()
        )

    def _human_iql_cutoff(self) -> int:
        done = self.human_example_buffer.data_done
        if done is None:
            raise RuntimeError("human_example done storage is unavailable")
        terminals = torch.nonzero(
            done[: self.human_example_buffer.tot_transition],
            as_tuple=False,
        ).flatten()
        required = int(self.config.human_iql_episode_count)
        if len(terminals) < required:
            raise RuntimeError(
                "human-IQL-grounded learner requires "
                f"{required} complete human success suffixes, found "
                f"{len(terminals)} under {self.output_dir / 'human_example'}"
            )
        return int(terminals[required - 1].detach().cpu().item()) + 1

    def _sample_human_positions(self, count: int) -> torch.Tensor:
        return self._exposure_balanced_positions(
            self.human_candidates, count, self.human_sample_counts
        )

    def _refresh_human_candidates(self) -> None:
        """Sample every accumulated complete human-to-success suffix.

        ``first20_cutoff`` is used only for startup IQL. Both the moving V_H
        IQL update and the SAC floor must use later human interventions too.
        """

        self.human_candidates = torch.arange(
            self.human_example_buffer.tot_transition,
            dtype=torch.long,
            device=self.human_example_buffer.config.data_device,
        )

    def _pretrain_human_iql(self) -> None:
        policy = self.policy
        generator = torch.Generator(
            device=self.human_example_buffer.config.data_device
        ).manual_seed(801)
        print(
            "[HUMAN IQL] independent all-horizon pretrain from "
            f"{self.config.human_iql_episode_count} episodes / "
            f"{self.first20_cutoff} transitions",
            flush=True,
        )
        for update in range(1, self.config.human_iql_updates + 1):
            positions = torch.randint(
                self.first20_cutoff,
                (self.config.human_example_batch_size,),
                generator=generator,
                device=self.human_example_buffer.config.data_device,
            )
            batch = self.human_example_buffer.sample_by_positions(
                positions, policy.device
            )
            metrics = policy._update_moving_human_iql(batch)

            if (
                update == 1
                or update % self.config.human_iql_log_every == 0
                or update == self.config.human_iql_updates
            ):
                for name, metric in metrics.items():
                    prefix = (
                        "Loss_IQL" if self.config.indexed_replay_views
                        else "pretrain_iql"
                    )
                    self.writer.add_scalar(
                        f"{prefix}/{name}",
                        torch.as_tensor(metric).float().mean().detach().cpu().item(),
                        update,
                    )
                print(
                    f"[HUMAN IQL] {update}/{self.config.human_iql_updates} "
                    f"Q={float(metrics['human_iql_critic_loss']):.5f} "
                    f"V={float(metrics['human_iql_loss']):.5f}",
                    flush=True,
                )

        self._warmup_q_only_actor()
        policy.finalize_human_iql()
        self._save_model_and_state()
        print("[HUMAN IQL] pretraining complete; checkpoint saved", flush=True)

    def _warmup_q_only_actor(self) -> None:
        policy = self.policy
        optimizer = torch.optim.Adam(
            [*policy.encoder_actor.parameters(), *policy.actor.parameters()],
            lr=policy.config.actor_lr,
        )
        for module in (
            policy.human_critic_encoder,
            policy.human_critic_1,
            policy.human_critic_2,
        ):
            module.requires_grad_(False)
        generator = torch.Generator(
            device=self.human_example_buffer.config.data_device
        ).manual_seed(901)
        for update in range(1, self.config.q_actor_warmup_updates + 1):
            positions = torch.randint(
                self.first20_cutoff,
                (self.config.human_example_batch_size,),
                generator=generator,
                device=self.human_example_buffer.config.data_device,
            )
            batch = self.human_example_buffer.sample_by_positions(
                positions, policy.device
            )
            obs = batch["observation"].float()
            action = policy.actor_actions(obs)
            feature = policy.human_critic_encoder(obs, detach=True)
            q = torch.minimum(
                policy.human_critic_1(feature, action),
                policy.human_critic_2(feature, action),
            )
            if policy.config.actor_horizons is None:
                # Match the online actor objective exactly: mean Q over all
                # remaining-horizon heads h=1..H.
                loss = -q.mean()
            else:
                indices = torch.as_tensor(
                    [horizon - 1 for horizon in policy.config.actor_horizons],
                    device=policy.device,
                )
                loss = -q[:, indices].mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*policy.encoder_actor.parameters(), *policy.actor.parameters()],
                10.0,
            )
            optimizer.step()
            if update == 1 or update % 500 == 0 or update == self.config.q_actor_warmup_updates:
                self.writer.add_scalar(
                    (
                        "Loss_SAC/actor_standard_Q"
                        if self.config.indexed_replay_views
                        else "pretrain_actor/q_only_loss"
                    ),
                    float(loss.detach()),
                    update,
                )
        for module in (
            policy.human_critic_encoder,
            policy.human_critic_1,
            policy.human_critic_2,
        ):
            module.requires_grad_(True)

    @torch.inference_mode()
    def _audit_human_iql_fit(self) -> dict[str, float]:
        policy = self.policy
        positions = torch.arange(
            self.first20_cutoff,
            device=self.human_example_buffer.config.data_device,
        )
        batch = self.human_example_buffer.sample_by_positions(
            positions, policy.device
        )
        obs = batch["observation"].float()
        action = batch["action"].float()
        value = policy.human_value(
            policy.human_value_encoder(obs, detach=False)
        )
        q = policy.human_critic_min(obs, action)
        task_reward = batch["reward"].float().reshape(-1)
        done = batch["done"].bool().reshape(-1)
        success = done & (task_reward > 0.0)
        reward = task_reward
        next_value = policy.human_value(
            policy.human_value_encoder(batch["next_observation"].float(), detach=False)
        )
        target, valid = build_vector_bellman_target(
            reward,
            success,
            torch.zeros_like(success),
            torch.ones_like(success),
            next_value,
            gamma=policy.config.gamma,
            lower_bound=policy.human_critic_1.lower_bound,
        )
        actor_action = policy.actor_actions(obs)
        actor_q = policy.human_critic_min(obs, actor_action)
        human_q = q
        return {
            "all_horizon_bellman_mae": float(
                (q - target).abs()[valid].mean().detach().cpu()
            ),
            "h1_bellman_mae": float(
                (q[:, 0] - target[:, 0]).abs().mean().detach().cpu()
            ),
            "value_rank_violation": float(
                horizon_extension_violation(
                    value, policy.config.stay_step_penalty
                ).mean().detach().cpu()
            ),
            "critic_rank_violation": float(
                horizon_extension_violation(
                    q, policy.config.stay_step_penalty
                ).mean().detach().cpu()
            ),
            "logged_q_minus_v": float((q - value).mean().detach().cpu()),
            "q_only_actor_minus_human_q": float(
                (actor_q - human_q).mean().detach().cpu()
            ),
            "q_only_actor_l2_from_human": float(
                torch.linalg.vector_norm(
                    actor_action - action, dim=1
                ).mean().detach().cpu()
            ),
        }

    def _exposure_balanced_positions(
        self,
        candidates: torch.Tensor,
        count: int,
        sample_counts: torch.Tensor,
    ) -> torch.Tensor:
        if len(candidates) == 0:
            raise RuntimeError("no valid replay candidates")
        exposure = sample_counts[candidates]
        inverse = (exposure + 1.0).pow(-self.config.replay_exposure_power)
        inverse /= inverse.sum().clamp_min(1e-12)
        uniform = torch.full_like(inverse, 1.0 / len(inverse))
        probability = (
            (1.0 - self.config.replay_uniform_mix) * inverse
            + self.config.replay_uniform_mix * uniform
        )
        selected = candidates[
            torch.multinomial(probability, int(count), replacement=True)
        ]
        sample_counts.index_add_(
            0,
            selected,
            torch.ones(len(selected), device=sample_counts.device),
        )
        return selected

    def _valid_online_positions(self) -> torch.Tensor:
        total = self.online_buffer.tot_transition
        safe_count = max(0, total - 1)
        safe = torch.arange(
            safe_count,
            device=self.online_buffer.config.data_device,
            dtype=torch.long,
        )
        done = self.online_buffer.data_done
        if done is None or safe_count >= total:
            return safe
        tail = torch.arange(
            safe_count,
            total,
            device=self.online_buffer.config.data_device,
            dtype=torch.long,
        )
        return torch.cat((safe, tail[done[tail]]))

    def _valid_offline_positions(self) -> torch.Tensor:
        if self.config.indexed_replay_views:
            raw_ids = self.offline_buffer.raw_ids
            if len(raw_ids) == 0:
                return raw_ids
            positions = self.online_buffer.raw_positions(raw_ids)
            done = self.online_buffer.data_done
            total = self.online_buffer.tot_transition
            return positions[(positions + 1 < total) | done[positions]]
        positions = self.offline_buffer.idx[: self.offline_buffer.tot_transition]
        if len(positions) == 0:
            return positions
        total = self.online_buffer.tot_transition
        done = self.online_buffer.data_done
        if done is None:
            raise RuntimeError("replay done storage is unavailable")
        return positions[(positions + 1 < total) | done[positions]]

    def _refresh_replay_candidates(self) -> None:
        """Refresh replay index caches after replay topology changes."""

        self.online_candidates = self._valid_online_positions()
        self.offline_candidates = self._valid_offline_positions()

    def _ensure_timeout_next_capacity(self, required: int) -> None:
        current = len(self.timeout_next_observations)
        if required <= current:
            return
        capacity = max(required, max(64, max(1, current) * 2))
        storage = torch.empty(
            (capacity, int(self.task.config.obs_dims)),
            dtype=torch.float32,
            device=self.online_buffer.config.data_device,
        )
        if self.timeout_next_count:
            storage[: self.timeout_next_count].copy_(
                self.timeout_next_observations[: self.timeout_next_count]
            )
        self.timeout_next_observations = storage

    def _register_timeout_next_observations(
        self,
        transitions: list[DictMessage],
        raw_start: int,
    ) -> None:
        offsets: list[int] = []
        states: list[DictMessage] = []
        infos: list[DictMessage] = []
        for offset, transition in enumerate(transitions):
            if not bool(transition.get("done", False)):
                continue
            if float(transition.get("reward", 0.0)) > 0.0:
                continue
            raw_next_obs = transition.get("raw_next_obs")
            if not isinstance(raw_next_obs, dict):
                continue
            offsets.append(offset)
            states.append(raw_next_obs)
            infos.append(transition["info"])
        if not states:
            return

        count = len(states)
        required = self.timeout_next_count + count
        self._ensure_timeout_next_capacity(required)
        offset_tensor = torch.as_tensor(
            offsets,
            dtype=torch.long,
            device=self.timeout_next_id.device,
        )
        build_observations = getattr(self.task, "build_observations", None)
        if build_observations is not None:
            encoded = build_observations(states, infos, augment=False)
        else:
            encoded = torch.stack(
                [
                    self.task.build_obs(state, info, augment=False)
                    for state, info in zip(states, infos, strict=True)
                ],
                dim=0,
            )
        encoded = encoded.detach().to(
            self.timeout_next_observations.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        row_start = self.timeout_next_count
        row_end = row_start + count
        self.timeout_next_observations[row_start:row_end].copy_(encoded)
        replay_positions = raw_start + offset_tensor
        self.timeout_next_id[replay_positions] = torch.arange(
            row_start,
            row_end,
            dtype=torch.long,
            device=self.timeout_next_id.device,
        )
        self.timeout_next_count = row_end
        self.has_timeout_next_rows = True

    def _rebuild_timeout_next_observations(self) -> None:
        rows = self._load_raw_replay(self.output_dir / "buffer")
        self._register_timeout_next_observations(rows, 0)

    def _patch_timeout_next_observation(
        self,
        batch: DictMessage,
        positions: torch.Tensor,
        target_device: torch.device | str,
    ) -> None:
        if not self.has_timeout_next_rows:
            batch["timeout_next_valid"] = torch.zeros(
                len(positions), dtype=torch.bool, device=target_device
            )
            return
        ids = self.timeout_next_id[positions]
        valid = ids >= 0
        target_valid = valid.to(target_device, non_blocking=True)
        selected = self.timeout_next_observations[ids.clamp_min(0)].to(
            target_device, non_blocking=True
        )
        batch["timeout_next_valid"] = target_valid
        batch["next_observation"] = torch.where(
            target_valid[:, None], selected, batch["next_observation"]
        )

    def on_actor_transitions(
        self,
        transitions: list[DictMessage],
        raw_ids: list[int] | None = None,
    ) -> None:
        suffix = self._complete_human_suffix(
            transitions, self.config.human_example_min_steps
        )
        if suffix:
            if self.config.indexed_replay_views:
                if raw_ids is None:
                    raise RuntimeError("indexed human suffix requires raw IDs")
                start = len(transitions) - len(suffix)
                self.human_example_buffer.add_many(raw_ids[start:])
            else:
                self.human_example_buffer.add_many(suffix)
            self.human_example_episodes += 1
            self._refresh_human_candidates()

    def step(self) -> None:
        self._maybe_periodic_save()
        transitions = self.transition_receiver.recv(None)
        if transitions is not None:
            raw_start = self.online_buffer.tot_transition
            for transition in transitions:
                self._log_actor_transition(transition)
            transition_indices = self.online_buffer.add_many(transitions)
            raw_ids = list(range(raw_start, raw_start + len(transitions)))
            self.on_actor_transitions(transitions, raw_ids)
            self._register_timeout_next_observations(transitions, raw_start)
            for offset, (transition, transition_idx) in enumerate(zip(
                transitions, transition_indices, strict=True
            )):
                if bool(transition["info"]["is_intervene"]):
                    self.offline_buffer.add(
                        raw_ids[offset]
                        if self.config.indexed_replay_views
                        else transition_idx
                    )
            self._refresh_replay_candidates()

        if self.online_buffer.tot_transition < max(
            2, int(self.config.transitions_before_start)
        ):
            return

        batch_size = self.config.batch_size
        offline_candidates = self.offline_candidates
        n_offline = min(
            max(1, int(batch_size * self.config.offline_batch_ratio)),
            len(offline_candidates),
        )
        n_online = max(1, batch_size - n_offline)
        online_positions = self._exposure_balanced_positions(
            self.online_candidates,
            n_online,
            self.replay_sample_counts,
        )
        if n_offline:
            offline_positions = self._exposure_balanced_positions(
                offline_candidates,
                n_offline,
                self.replay_sample_counts,
            )
            positions = torch.cat((online_positions, offline_positions))
        else:
            positions = online_positions
        device = self.policy.device
        batch = self.online_buffer.sample_by_positions(positions, device)
        self._patch_timeout_next_observation(batch, positions, device)

        # Algorithm-specific gates must classify the exact replay observation,
        # not the noisy tensor used to regularize the trainable networks.
        batch["gate_observation"] = batch["observation"].clone()
        batch["gate_next_observation"] = batch["next_observation"].clone()

        batch["observation"] = batch["observation"] + (
            torch.randn_like(batch["observation"])
            * self.config.observation_noise_std
        )
        batch["action"] = (
            batch["action"]
            + torch.randn_like(batch["action"]) * self.config.action_noise_std
        ).clamp_(-1.0, 1.0)

        human_positions = self._sample_human_positions(
            self.config.human_example_batch_size
        )
        batch["human_example"] = self.human_example_buffer.sample_by_positions(
            human_positions, device
        )
        result = self.policy.update(batch=batch)
        self._accumulate_train_metrics(result)
        self.train_steps += 1

        if self.train_steps % self.config.send_parameters_every == 0:
            self.parameters_sender.send(self._runtime_policy_export())
        if self.train_steps % self.config.log_every == 0:
            self._flush_train_metrics()
            self.writer.add_scalar(
                "buffer/human_example_episodes",
                float(self.human_example_episodes),
                self.train_steps,
            )
            self.writer.add_scalar(
                "buffer/human_example_transitions",
                float(self.human_example_buffer.tot_transition),
                self.train_steps,
            )
            print(
                f"[HUMAN IQL GROUNDED] step={self.train_steps}",
                flush=True,
            )

    def run(self) -> None:
        try:
            print("Human-IQL-grounded learner started", flush=True)
            # The physical actor blocks until the fully initialized Q-only
            # policy is available.
            self.parameters_sender.send(self._runtime_policy_export())
            while True:
                self.step()
        except KeyboardInterrupt:
            print("KeyboardInterrupted", flush=True)
        finally:
            self._flush_train_metrics()
            self._save_model_and_state()
            self._checkpoint_writer.close()
            self.writer.flush()
            self.writer.close()
