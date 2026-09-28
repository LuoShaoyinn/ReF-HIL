from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from .observation_encoder import ObservationEncoder
from shared.actor_network import ACTOR_SNAPSHOT_FORMAT, ActorNetworkConfig
from shared.policy import BasePolicy, PolicyConfig
from shared.zmq import DictMessage

from .iql_modules import (
    BoundedVectorHead,
    BilinearVectorCritic,
    DeterministicActorHead,
    build_vector_bellman_target,
    expectile_loss,
    horizon_extension_violation,
    horizon_lower_bound,
    horizon_ranking_loss,
)


MONITOR_HORIZONS = (1, 5, 10, 20, 50, 100)


@dataclass(kw_only=True)
class HumanIQLGroundedPolicyConfig(PolicyConfig):
    obs_dims: int = 1552
    action_dims: int = 7
    mechanism_obs_dims: int = 16
    encoder_dim: int = 256
    hidden_dim: int = 256
    horizons: int = 100
    bilinear_rank: int = 32

    gamma: float = 0.99
    target_tau: float = 0.01
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    utd_ratio: int = 1
    minimum_step_reward: float = -0.06

    human_floor_weight: float = 1.0
    human_iql_expectile: float = 0.75
    human_iql_critic_lr: float = 1e-4
    human_iql_value_lr: float = 1e-4
    stay_step_penalty: float = 0.01
    horizon_rank_weight: float = 0.1
    td_mse_weight: float = 0.5
    # ``None`` is the mathematically complete actor objective: the uniform
    # mean over every learned remaining-horizon head h=1..H.  A non-empty
    # tuple remains available only for a deliberate ablation.
    actor_horizons: tuple[int, ...] | None = None


class HumanIQLGroundedPolicy(BasePolicy):
    """Q-only actor-critic initialized and grounded by human-suffix IQL.

    The actor and its encoder start from independent random parameters. They
    never receive behavior-cloning or action-label gradients.
    """

    config: HumanIQLGroundedPolicyConfig

    def __init__(self, config: HumanIQLGroundedPolicyConfig) -> None:
        # Enable TF32-capable float32 matmuls where the backend supports them.
        # Losses, Bellman targets, and optimizer state remain float32.
        torch.set_float32_matmul_precision("high")
        super().__init__(config=config)
        self.config = config
        self.device = torch.device(config.device)
        if config.action_dims < 1:
            raise ValueError("human-IQL-grounded policy requires continuous actions")
        if config.horizons < 1:
            raise ValueError("horizons must be positive")
        if config.utd_ratio < 1:
            raise ValueError("utd_ratio must be positive")
        if not 0.5 < config.human_iql_expectile < 1.0:
            raise ValueError("human_iql_expectile must be in (0.5, 1.0)")
        if config.actor_horizons is not None and (
            not config.actor_horizons
            or any(
                horizon < 1 or horizon > config.horizons
                for horizon in config.actor_horizons
            )
        ):
            raise ValueError("actor_horizons must lie within vector Q")
        for name in (
            "human_floor_weight",
            "stay_step_penalty",
            "horizon_rank_weight",
            "td_mse_weight",
        ):
            if float(getattr(config, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

        self.update_step = 0
        self.human_iql_initialized = False
        self.loaded_legacy_iql = False
        self._build_networks()
        self._build_optimizers()

    def _build_networks(self) -> None:
        config = self.config
        lower = horizon_lower_bound(
            config.horizons,
            config.gamma,
            config.minimum_step_reward,
        )
        self.encoder_critic = ObservationEncoder(
            config.obs_dims,
            config.mechanism_obs_dims,
            config.encoder_dim,
        ).to(self.device)
        self.encoder_actor = ObservationEncoder(
            config.obs_dims,
            config.mechanism_obs_dims,
            config.encoder_dim,
        ).to(self.device)
        feature_dim = int(self.encoder_critic.output_dim)
        critic_args = (
            feature_dim,
            config.action_dims,
            config.hidden_dim,
            lower,
            config.bilinear_rank,
        )
        self.critic_1 = BilinearVectorCritic(*critic_args).to(self.device)
        self.critic_2 = BilinearVectorCritic(*critic_args).to(self.device)
        # A zero Q/V output is a stable origin for offline Bellman propagation.
        nn.init.zeros_(self.critic_1.output.weight)
        nn.init.zeros_(self.critic_1.output.bias)
        nn.init.zeros_(self.critic_2.output.weight)
        nn.init.zeros_(self.critic_2.output.bias)
        self.encoder_target = copy.deepcopy(self.encoder_critic).to(self.device)
        self.critic_target_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.critic_target_2 = copy.deepcopy(self.critic_2).to(self.device)

        self.actor = DeterministicActorHead(
            int(self.encoder_actor.output_dim), config.hidden_dim, config.action_dims
        ).to(self.device)

        self.human_value_encoder = ObservationEncoder(
            config.obs_dims,
            config.mechanism_obs_dims,
            config.encoder_dim,
        ).to(self.device)
        self.human_value = BoundedVectorHead(
            int(self.human_value_encoder.output_dim),
            config.hidden_dim,
            lower,
        ).to(self.device)
        value_output = cast(nn.Linear, self.human_value.net[-1])
        nn.init.zeros_(value_output.weight)
        nn.init.zeros_(value_output.bias)

        self.human_critic_encoder = ObservationEncoder(
            config.obs_dims,
            config.mechanism_obs_dims,
            config.encoder_dim,
        ).to(self.device)
        human_feature_dim = int(self.human_critic_encoder.output_dim)
        human_critic_args = (
            human_feature_dim,
            config.action_dims,
            config.hidden_dim,
            lower,
            config.bilinear_rank,
        )
        self.human_critic_1 = BilinearVectorCritic(*human_critic_args).to(
            self.device
        )
        self.human_critic_2 = BilinearVectorCritic(*human_critic_args).to(
            self.device
        )
        for critic in (self.human_critic_1, self.human_critic_2):
            nn.init.zeros_(critic.output.weight)
            nn.init.zeros_(critic.output.bias)

        for module in (
            self.encoder_target,
            self.critic_target_1,
            self.critic_target_2,
        ):
            module.requires_grad_(False)
            module.eval()

        self.networks: dict[str, nn.Module] = {
            "encoder_critic": self.encoder_critic,
            "encoder_target": self.encoder_target,
            "critic_1": self.critic_1,
            "critic_2": self.critic_2,
            "critic_target_1": self.critic_target_1,
            "critic_target_2": self.critic_target_2,
            "encoder_actor": self.encoder_actor,
            "actor": self.actor,
            "human_value_encoder": self.human_value_encoder,
            "human_value": self.human_value,
            "human_critic_encoder": self.human_critic_encoder,
            "human_critic_1": self.human_critic_1,
            "human_critic_2": self.human_critic_2,
        }

    def _build_optimizers(self) -> None:
        self.critic_optim = torch.optim.Adam(
            [
                *self.encoder_critic.parameters(),
                *self.critic_1.parameters(),
                *self.critic_2.parameters(),
            ],
            lr=self.config.critic_lr,
        )
        self.actor_optim = torch.optim.Adam(
            [*self.encoder_actor.parameters(), *self.actor.parameters()],
            lr=self.config.actor_lr,
        )
        self.human_value_optim = torch.optim.Adam(
            [
                *self.human_value_encoder.parameters(),
                *self.human_value.parameters(),
            ],
            lr=self.config.human_iql_value_lr,
        )
        self.human_critic_optim = torch.optim.Adam(
            [
                *self.human_critic_encoder.parameters(),
                *self.human_critic_1.parameters(),
                *self.human_critic_2.parameters(),
            ],
            lr=self.config.human_iql_critic_lr,
        )
        self.optimizers = {
            "critic_optimizer": self.critic_optim,
            "actor_optimizer": self.actor_optim,
            "human_value_optimizer": self.human_value_optim,
            "human_critic_optimizer": self.human_critic_optim,
        }

    def finalize_human_iql(self) -> None:
        # SAC starts from the independently learned human IQL critic. After
        # this one-time copy, SAC and IQL have disjoint parameters and
        # optimizers; only detached V_H is allowed to constrain SAC.
        self.encoder_critic.load_state_dict(
            self.human_critic_encoder.state_dict()
        )
        self.critic_1.load_state_dict(self.human_critic_1.state_dict())
        self.critic_2.load_state_dict(self.human_critic_2.state_dict())
        self.encoder_target.load_state_dict(self.encoder_critic.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        for module in (
            self.encoder_target,
            self.critic_target_1,
            self.critic_target_2,
        ):
            module.requires_grad_(False)
            module.eval()
        # Q_IQL and V_H remain an independent moving IQL system. They are
        # never updated by SAC TD, actor, floor, or outside-field losses.
        for module in (
            self.human_critic_encoder,
            self.human_critic_1,
            self.human_critic_2,
            self.human_value_encoder,
            self.human_value,
        ):
            module.requires_grad_(True)
            module.train()
        self.human_iql_initialized = True
        # Start online optimization with fresh moments while preserving weights.
        self._build_optimizers()

    def actor_actions(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.encoder_actor(observation, detach=False)
        return self.actor(feature)

    def critic_min(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        *,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        feature = self.encoder_critic(observation, detach=detach_encoder)
        return torch.minimum(
            self.critic_1(feature, action),
            self.critic_2(feature, action),
        )

    def human_critic_min(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        *,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        feature = self.human_critic_encoder(
            observation, detach=detach_encoder
        )
        return torch.minimum(
            self.human_critic_1(feature, action),
            self.human_critic_2(feature, action),
        )

    def sample_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del deterministic
        obs = observation.to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor_actions(obs)
        return action.squeeze(0)

    def _human_floor(
        self,
        human_batch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=self.device)
        if human_batch is None:
            return zero, {
                "human_floor_loss": zero,
                "human_floor_shortfall": zero,
                "human_floor_violation_fraction": zero,
                "human_floor_pairs": zero,
            }
        obs = cast(torch.Tensor, human_batch["observation"]).to(
            self.device
        ).float()
        action = cast(torch.Tensor, human_batch["action"]).to(
            self.device
        ).float()
        feature = self.encoder_critic(obs, detach=False)
        q1 = self.critic_1(feature, action)
        q2 = self.critic_2(feature, action)
        q_min = torch.minimum(q1, q2)
        view_count = int(human_batch.get("view_count", 1))
        with torch.no_grad():
            value = self.human_value(
                self.human_value_encoder(obs, detach=False)
            )
            if view_count > 1:
                value = value.reshape(
                    -1, view_count, self.config.horizons
                ).mean(dim=1, keepdim=True).expand(
                    -1, view_count, -1
                ).reshape(-1, self.config.horizons)
        shortfall = torch.relu(value - q_min)
        loss = shortfall.square().mean()
        metrics: dict[str, torch.Tensor] = {
            "human_floor_loss": loss.detach(),
            "human_floor_shortfall": shortfall.mean().detach(),
            "human_floor_violation_fraction": (
                shortfall > 0.0
            ).float().mean().detach(),
            "human_floor_pairs": torch.as_tensor(
                float(shortfall.numel()), device=self.device
            ),
            "human_value_rank_violation": horizon_extension_violation(
                value, self.config.stay_step_penalty
            ).mean().detach(),
            "human_q_rank_violation": horizon_extension_violation(
                q_min, self.config.stay_step_penalty
            ).mean().detach(),
        }
        for horizon in MONITOR_HORIZONS:
            if horizon > self.config.horizons:
                continue
            index = horizon - 1
            metrics[f"human_v_h{horizon}"] = value[:, index].mean().detach()
            metrics[f"human_q_h{horizon}"] = q_min[:, index].mean().detach()
            metrics[f"human_shortfall_h{horizon}"] = (
                shortfall[:, index].mean().detach()
            )
        return loss, metrics

    def _update_moving_human_iql(
        self,
        human_batch: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        """Update an independent vector IQL system on human suffixes.

        Every transition supervises every horizon. Q_IQL uses the homogeneous
        Bellman shift ``y_1=r`` and ``y_h=r+gamma*V_H(s',h-1)``. V_H is the
        expectile projection of detached Q_IQL, never of SAC Q.
        """

        zero = torch.zeros((), device=self.device)
        if human_batch is None:
            return {
                "human_iql_loss": zero,
                "human_iql_critic_loss": zero,
                "human_iql_td_smooth_loss": zero,
                "human_iql_td_mse_loss": zero,
                "human_iql_expectile_loss": zero,
                "human_iql_rank_loss": zero,
                "human_iql_q_minus_v": zero,
            }
        obs = cast(torch.Tensor, human_batch["observation"]).to(self.device).float()
        next_obs = cast(torch.Tensor, human_batch["next_observation"]).to(
            self.device
        ).float()
        action = cast(torch.Tensor, human_batch["action"]).to(self.device).float()
        task_reward = cast(torch.Tensor, human_batch["reward"]).to(
            self.device
        ).float().reshape(-1)
        done = cast(torch.Tensor, human_batch["done"]).to(
            self.device
        ).bool().reshape(-1)
        success = done & (task_reward > 0.0)
        reward = task_reward
        view_count = int(human_batch.get("view_count", 1))

        # Independent IQL Bellman update. Successful suffixes have no failure
        # timeout; every h is valid and receives a target on every transition.
        with torch.no_grad():
            next_value = self.human_value(
                self.human_value_encoder(next_obs, detach=False)
            )
            if "unrecoverable_next" in human_batch:
                from .unrecoverable import failure_vector
                failed = human_batch["unrecoverable_next"].to(self.device).bool()
                no_recovery = failure_vector(self.config.horizons, self.config.gamma,
                    self.config.stay_step_penalty, device=self.device)
                next_value = torch.where(failed[:, None], no_recovery, next_value)
            if view_count > 1:
                next_value = next_value.reshape(
                    -1, view_count, self.config.horizons
                ).mean(dim=1, keepdim=True).expand(
                    -1, view_count, -1
                ).reshape(-1, self.config.horizons)
            q_target, q_valid = build_vector_bellman_target(
                reward,
                success,
                torch.zeros_like(success),
                torch.ones_like(success),
                next_value,
                gamma=self.config.gamma,
                lower_bound=self.human_critic_1.lower_bound,
            )
        human_feature = self.human_critic_encoder(obs, detach=False)
        human_q1 = self.human_critic_1(human_feature, action)
        human_q2 = self.human_critic_2(human_feature, action)
        valid_float = q_valid.float()
        denominator = valid_float.sum().clamp_min(1.0)
        q_smooth = (
            F.smooth_l1_loss(
                human_q1, q_target, beta=0.05, reduction="none"
            )
            + F.smooth_l1_loss(
                human_q2, q_target, beta=0.05, reduction="none"
            )
        )
        q_smooth = (q_smooth * valid_float).sum() / denominator
        q_mse = (
            (human_q1 - q_target).square()
            + (human_q2 - q_target).square()
        )
        q_mse = (q_mse * valid_float).sum() / denominator
        q_rank = horizon_ranking_loss(
            human_q1, self.config.stay_step_penalty
        ) + horizon_ranking_loss(
            human_q2, self.config.stay_step_penalty
        )
        q_loss = (
            q_smooth
            + self.config.td_mse_weight * q_mse
            + self.config.horizon_rank_weight * q_rank
        )
        self.human_critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                *self.human_critic_encoder.parameters(),
                *self.human_critic_1.parameters(),
                *self.human_critic_2.parameters(),
            ],
            10.0,
        )
        self.human_critic_optim.step()

        # V_H is projected from Q_IQL only. Grouped observation views share a
        # physical-state target, while all view predictions receive gradients.
        with torch.no_grad():
            q_value_target = self.human_critic_min(obs, action)
            if view_count > 1:
                q_value_target = q_value_target.reshape(
                    -1, view_count, self.config.horizons
                ).mean(dim=1, keepdim=True).expand(
                    -1, view_count, -1
                ).reshape(-1, self.config.horizons)
        value = self.human_value(self.human_value_encoder(obs, detach=False))
        expectile = expectile_loss(
            q_value_target - value,
            self.config.human_iql_expectile,
        )
        rank = horizon_ranking_loss(value, self.config.stay_step_penalty)
        consistency = torch.zeros((), device=self.device)
        augmentation_std = torch.zeros((), device=self.device)
        if view_count > 1:
            grouped_value = value.reshape(
                -1, view_count, self.config.horizons
            )
            consistency = (
                grouped_value - grouped_value.mean(dim=1, keepdim=True)
            ).square().mean()
            augmentation_std = grouped_value.std(
                dim=1, correction=0
            ).mean()
        loss = (
            expectile
            + self.config.horizon_rank_weight * rank
            + float(getattr(self.config, "human_value_consistency_weight", 0.0))
            * consistency
        )
        self.human_value_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*self.human_value_encoder.parameters(), *self.human_value.parameters()],
            10.0,
        )
        self.human_value_optim.step()
        metrics = {
            "human_iql_loss": loss.detach(),
            "human_iql_critic_loss": q_loss.detach(),
            "human_iql_td_smooth_loss": q_smooth.detach(),
            "human_iql_td_mse_loss": q_mse.detach(),
            "human_iql_critic_rank_loss": q_rank.detach(),
            "human_iql_expectile_loss": expectile.detach(),
            "human_iql_rank_loss": rank.detach(),
            "human_iql_q_minus_v": (
                q_value_target - value
            ).mean().detach(),
            "human_iql_consistency": consistency.detach(),
            "human_value_augmentation_std": augmentation_std.detach(),
            "human_iql_valid_pairs": denominator.detach(),
        }
        human_q_min = torch.minimum(human_q1, human_q2)
        for horizon in MONITOR_HORIZONS:
            if horizon <= self.config.horizons:
                metrics[f"human_iql_q_h{horizon}"] = (
                    human_q_min[:, horizon - 1].mean().detach()
                )
        return metrics

    def _critic_loss(
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        success: torch.Tensor,
        timeout: torch.Tensor,
        timeout_next_valid: torch.Tensor,
        next_obs: torch.Tensor,
        human_batch: dict[str, torch.Tensor] | None,
        unrecoverable_next: torch.Tensor | None = None,
        unrecoverable_observation: torch.Tensor | None = None,
        gate_obs: torch.Tensor | None = None,
        gate_next_obs: torch.Tensor | None = None,
        view_count: int = 1,
        ood_incoming_mask: torch.Tensor | None = None,
        correction_batch: dict[str, torch.Tensor] | None = None,
        certified_suffix_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del (
            gate_obs,
            gate_next_obs,
            view_count,
            ood_incoming_mask,
            correction_batch,
            certified_suffix_next,
        )
        feature = self.encoder_critic(obs, detach=False)
        q1 = self.critic_1(feature, action)
        q2 = self.critic_2(feature, action)
        with torch.no_grad():
            next_action = self.actor_actions(next_obs)
            next_feature = self.encoder_target(next_obs, detach=False)
            next_q = torch.minimum(
                self.critic_target_1(next_feature, next_action),
                self.critic_target_2(next_feature, next_action),
            )
            target, valid = build_vector_bellman_target(
                reward,
                success,
                timeout,
                timeout_next_valid,
                next_q,
                gamma=self.config.gamma,
                lower_bound=self.critic_1.lower_bound,
            )
        float_valid = valid.float()
        denominator = float_valid.sum().clamp_min(1.0)
        smooth = F.smooth_l1_loss(
            q1, target, beta=0.05, reduction="none"
        ) + F.smooth_l1_loss(q2, target, beta=0.05, reduction="none")
        smooth = (smooth * float_valid).sum() / denominator
        mse = (q1 - target).square() + (q2 - target).square()
        mse = (mse * float_valid).sum() / denominator
        rank = horizon_ranking_loss(
            q1, self.config.stay_step_penalty
        ) + horizon_ranking_loss(q2, self.config.stay_step_penalty)
        floor, floor_metrics = self._human_floor(human_batch)
        total = (
            smooth
            + self.config.td_mse_weight * mse
            + self.config.horizon_rank_weight * rank
            + self.config.human_floor_weight * floor
        )
        metrics = {
            "critic_td_smooth_loss": smooth.detach(),
            "critic_td_mse_loss": mse.detach(),
            "critic_horizon_rank_loss": rank.detach(),
            "critic_horizon_violation": (
                horizon_extension_violation(
                    torch.minimum(q1, q2), self.config.stay_step_penalty
                )
                .mean()
                .detach()
            ),
            "critic_valid_pairs": denominator.detach(),
            "timeout_fraction": timeout.float().mean().detach(),
            "timeout_next_valid_fraction": (timeout_next_valid.float().mean().detach()),
            **floor_metrics,
        }
        return total, metrics

    def _actor_loss(
        self,
        obs: torch.Tensor,
        gate_obs: torch.Tensor | None = None,
        view_count: int = 1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del gate_obs, view_count
        action = self.actor_actions(obs)
        feature = self.encoder_critic(obs, detach=True)
        q1 = self.critic_1(feature, action)
        q2 = self.critic_2(feature, action)
        q_min = torch.minimum(q1, q2)
        if self.config.actor_horizons is None:
            # E_{h ~ Uniform{1,...,H}} Q_h(s, pi(s)).  h=0 is not a
            # prediction head; it is the fixed Bellman boundary V_0=0.
            selected = q_min.mean(dim=1)
        else:
            indices = torch.as_tensor(
                [horizon - 1 for horizon in self.config.actor_horizons],
                device=self.device,
            )
            selected = q_min[:, indices].mean(dim=1)
        loss = -selected.mean()
        return loss, {
            "actor_q": selected.mean().detach(),
            "actor_action_norm": torch.linalg.vector_norm(
                action, dim=1
            ).mean().detach(),
            "actor_gripper_command_mean": action[:, -1].mean().detach(),
        }

    def update(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not self.human_iql_initialized:
            raise RuntimeError("human IQL must initialize the critic before update")
        obs = cast(torch.Tensor, batch["observation"]).to(self.device).float()
        next_obs = cast(torch.Tensor, batch["next_observation"]).to(self.device).float()
        action = cast(torch.Tensor, batch["action"]).to(self.device).float()
        task_reward = (
            cast(torch.Tensor, batch["reward"]).to(self.device).float().reshape(-1)
        )
        done = cast(torch.Tensor, batch["done"]).to(self.device).bool().reshape(-1)
        success = done & (task_reward > 0.0)
        timeout = done & ~success
        combined_reward = task_reward
        # Preserve the task's graded terminal success reward on every h.
        timeout_next_valid = (
            cast(
                torch.Tensor,
                batch.get(
                    "timeout_next_valid",
                    torch.zeros_like(timeout),
                ),
            )
            .to(self.device)
            .bool()
            .reshape(-1)
        )
        human_batch = cast(
            dict[str, torch.Tensor] | None,
            batch.get("human_example"),
        )

        critic_loss, metrics = self._critic_loss(
            obs=obs,
            action=action,
            reward=combined_reward,
            success=success,
            timeout=timeout,
            timeout_next_valid=timeout_next_valid,
            next_obs=next_obs,
            human_batch=human_batch,
            unrecoverable_next=batch.get('unrecoverable_next'),
            unrecoverable_observation=batch.get('unrecoverable_observation'),
            gate_obs=cast(torch.Tensor | None, batch.get("gate_observation")),
            gate_next_obs=cast(torch.Tensor | None, batch.get("gate_next_observation")),
            view_count=int(batch.get("view_count", 1)),
            ood_incoming_mask=cast(torch.Tensor | None, batch.get("ood_incoming_mask")),
            correction_batch=cast(
                dict[str, torch.Tensor] | None,
                batch.get("correction_example"),
            ),
            certified_suffix_next=cast(
                torch.Tensor | None,
                batch.get("certified_suffix_next"),
            ),
        )
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                *self.encoder_critic.parameters(),
                *self.critic_1.parameters(),
                *self.critic_2.parameters(),
            ],
            10.0,
        )
        self.critic_optim.step()
        # The floor used the previous detached V_H. This separate step moves
        # V_H for the next update, with no gradient into SAC Q.
        metrics.update(self._update_moving_human_iql(human_batch))
        from .unrecoverable import update_iql_failure
        metrics.update(update_iql_failure(self, batch.get('unrecoverable_observation'),
                                         batch.get('failure_value_transitions')))
        metrics["critic_loss"] = critic_loss.detach()
        metrics["mean_task_reward"] = task_reward.mean().detach()
        metrics["mean_combined_reward"] = combined_reward.mean().detach()

        next_step = self.update_step + 1
        if next_step % self.config.utd_ratio == 0:
            for module in (
                self.encoder_critic,
                self.critic_1,
                self.critic_2,
            ):
                module.requires_grad_(False)
            actor_loss, actor_metrics = self._actor_loss(
                obs,
                cast(torch.Tensor | None, batch.get("gate_observation")),
                int(batch.get("view_count", 1)),
            )
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*self.encoder_actor.parameters(), *self.actor.parameters()],
                10.0,
            )
            self.actor_optim.step()
            for module in (
                self.encoder_critic,
                self.critic_1,
                self.critic_2,
            ):
                module.requires_grad_(True)
            metrics["actor_loss"] = actor_loss.detach()
            metrics.update(actor_metrics)

        with torch.no_grad():
            for source, target in (
                (self.encoder_critic, self.encoder_target),
                (self.critic_1, self.critic_target_1),
                (self.critic_2, self.critic_target_2),
            ):
                for source_parameter, target_parameter in zip(
                    source.parameters(), target.parameters(), strict=True
                ):
                    target_parameter.lerp_(source_parameter, self.config.target_tau)
        self.update_step = next_step
        metrics["update_step"] = torch.as_tensor(
            float(self.update_step), device=self.device
        )
        return metrics

    def export(
        self,
        *,
        exclude_networks: frozenset[str] = frozenset(),
    ) -> DictMessage:
        state: DictMessage = {
            "algorithm": "human_iql_grounded_actor_critic_v2_independent_iql",
            "update_step": self.update_step,
            "human_iql_initialized": self.human_iql_initialized,
            "training_state": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
        }
        for name, module in self.networks.items():
            if name in exclude_networks:
                continue
            state[name] = {
                key: value.detach().cpu().numpy()
                for key, value in module.state_dict().items()
            }
        return state

    def export_actor(self) -> DictMessage:
        """Export only the algorithm-independent physical actor."""

        actor_config = ActorNetworkConfig(
            obs_dims=self.config.obs_dims,
            mechanism_obs_dims=self.config.mechanism_obs_dims,
            encoder_dim=self.config.encoder_dim,
            hidden_dim=self.config.hidden_dim,
            action_dims=self.config.action_dims,
        )
        return {
            "format": ACTOR_SNAPSHOT_FORMAT,
            "config": actor_config.as_dict(),
            "encoder": {
                key: value.detach().cpu().numpy()
                for key, value in self.encoder_actor.state_dict().items()
            },
            "actor": {
                key: value.detach().cpu().numpy()
                for key, value in self.actor.state_dict().items()
            },
            "output_activation": self.actor.output_activation,
        }

    def load(self, state: dict[str, Any]) -> None:
        algorithm = state.get("algorithm")
        if algorithm not in (
            None,
            "human_iql_grounded_actor_critic_v1",
            "human_iql_grounded_actor_critic_v2_independent_iql",
        ):
            raise ValueError(f"incompatible policy snapshot: {algorithm!r}")
        for name, module in self.networks.items():
            module_state = state.get(name)
            if not isinstance(module_state, dict):
                continue
            module.load_state_dict(
                {
                    key: torch.as_tensor(value, device=self.device)
                    for key, value in module_state.items()
                },
                strict=True,
            )
        if not all(
            isinstance(state.get(name), dict)
            for name in (
                "human_critic_encoder",
                "human_critic_1",
                "human_critic_2",
            )
        ):
            # Legacy snapshots used SAC Q as Q_IQL. Migrate once by copying
            # the saved SAC critic; subsequent optimizers are fully disjoint.
            self.human_critic_encoder.load_state_dict(
                self.encoder_critic.state_dict()
            )
            self.human_critic_1.load_state_dict(self.critic_1.state_dict())
            self.human_critic_2.load_state_dict(self.critic_2.state_dict())
            self.loaded_legacy_iql = True
        self.update_step = int(state.get("update_step", self.update_step))
        self.human_iql_initialized = bool(
            state.get("human_iql_initialized", self.human_iql_initialized)
        )
        if self.human_iql_initialized:
            for module in (
                self.human_critic_encoder,
                self.human_critic_1,
                self.human_critic_2,
                self.human_value_encoder,
                self.human_value,
            ):
                module.requires_grad_(True)
                module.train()
        training_state = state.get("training_state")
        if not isinstance(training_state, dict):
            return
        for name, optimizer in self.optimizers.items():
            optimizer_state = training_state.get(name)
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)
