from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F

from shared.zmq import DictMessage

from .action_proximity import (
    ACTION_LIKENESS_SIGMA,
    ActionProximityConfig,
    HumanActionProximity,
    RecoveryActorHead,
    gaussian_density_peak,
)
from .iql_floor import (
    MONITOR_HORIZONS,
    HumanIQLGroundedPolicy,
    HumanIQLGroundedPolicyConfig,
)
from .observation_encoder import ObservationEncoder
from .constrained_critic import ConstrainedCritic, RawObservation
from .constrained_learning import ConstrainedLearning
from .actor_optimization import native_cuda


@dataclass(kw_only=True)
class LimitActionPolicyConfig(HumanIQLGroundedPolicyConfig):
    actor_lr: float = 5e-4
    optimize_actor: bool = True
    compile_actor: bool = True
    fused_actor_adam: bool = True
    hard_iql_reference_floor: bool = False
    suffix_only_reference_floor: bool = True
    # Keep the human-suffix estimate mildly optimistic without the severe
    # recursive bootstrap amplification measured at expectiles 0.9 and 0.99.
    human_iql_expectile: float = 0.75
    recovery_actor_lr: float = 1e-4
    recovery_actor_beta: float = 200.0
    recovery_actor_max_weight: float = 100.0
    recovery_consistency_weight: float = 1.0
    actor_consistency_weight: float = 1.0
    human_floor_beta: float = 0.02
    human_floor_violation_tolerance: float = 0.01

    # Derived from mean episode-held-out demo BC error and Gaussian density by
    # LimitActionLearner before the fence is queried.
    action_likeness_threshold: float | None = None
    action_likeness_sigma: float = ACTION_LIKENESS_SIGMA
    action_likeness_reference_ratio: float | None = 0.9
    action_likeness_lr: float = 3e-4
    action_likeness_weight_decay: float = 1e-5
    correction_rank_weight: float = 2.0
    rejected_advantage_rank_weight: float = 2.0
    correction_noise_samples: int = 8
    correction_noise_std: float = 0.15
    correction_reference_exclusion_radius: float = 0.05
    # This weaker floor covers much more replay than the weight-1 suffix floor;
    # it prevents factual TD from creating an immediate value cliff outside
    # exact suffix rows without treating extrapolated IQL values as equally
    # certified.
    replay_reference_floor_weight: float = 0.10


class LimitActionPolicy(ConstrainedLearning, HumanIQLGroundedPolicy):
    """Reference-centered SAC with an analytical human-action fence.

    The inherited moving human IQL and human floor are retained. The
    BC-disagreement state checker, state gate, OOD replay view, and IQL runtime
    fallback are intentionally absent.
    """

    config: LimitActionPolicyConfig

    def _action_likeness_threshold(self) -> float:
        threshold = self.config.action_likeness_threshold
        if threshold is None:
            raise RuntimeError(
                "action-likeness threshold has not been calibrated by the learner"
            )
        return float(threshold)

    @torch.no_grad()
    def _fence_threshold(
        self, observation: torch.Tensor, reference_action: torch.Tensor
    ) -> torch.Tensor:
        ratio = self.config.action_likeness_reference_ratio
        if ratio is None:
            return observation.new_full(
                (len(observation),), self._action_likeness_threshold()
            )
        return ratio * self.action_likeness(observation, reference_action)

    def __init__(self, config: LimitActionPolicyConfig) -> None:
        if (
            config.action_likeness_reference_ratio is not None
            and not 0 < config.action_likeness_reference_ratio <= 1
        ):
            raise ValueError("action_likeness_reference_ratio must be in (0,1]")
        density_peak = gaussian_density_peak(
            config.action_dims, sigma=config.action_likeness_sigma
        )
        if config.action_likeness_threshold is not None and not (
            0.0 < config.action_likeness_threshold <= density_peak
        ):
            raise ValueError(
                "action_likeness_threshold must be in (0, Gaussian density peak]"
            )
        if config.action_likeness_reference_ratio is None or not (
            0.0 < config.action_likeness_reference_ratio <= 1.0
        ):
            raise ValueError("constrained Q requires a relative fence ratio in (0, 1]")
        if config.actor_horizons is not None:
            raise ValueError("constrained actor must optimize every horizon")
        if config.action_likeness_lr <= 0.0:
            raise ValueError("action_likeness_lr must be positive")
        if config.action_likeness_weight_decay < 0.0:
            raise ValueError("action_likeness_weight_decay must be non-negative")
        if config.correction_rank_weight < 0.0:
            raise ValueError("correction_rank_weight must be non-negative")
        if config.rejected_advantage_rank_weight < 0.0:
            raise ValueError("rejected_advantage_rank_weight must be non-negative")
        if config.correction_noise_samples < 0:
            raise ValueError("correction_noise_samples must be non-negative")
        if config.correction_noise_std < 0.0:
            raise ValueError("correction_noise_std must be non-negative")
        if config.correction_reference_exclusion_radius < 0.0:
            raise ValueError(
                "correction_reference_exclusion_radius must be non-negative"
            )
        if config.replay_reference_floor_weight < 0.0:
            raise ValueError("replay_reference_floor_weight must be non-negative")
        if config.recovery_actor_beta < 0.0:
            raise ValueError("recovery_actor_beta must be non-negative")
        if config.recovery_actor_max_weight < 1.0:
            raise ValueError("recovery_actor_max_weight must be at least one")
        if config.human_floor_beta <= 0.0:
            raise ValueError("human_floor_beta must be positive")
        if config.human_floor_violation_tolerance < 0.0:
            raise ValueError("human_floor_violation_tolerance must be non-negative")
        super().__init__(config)
        self.recovery_initialized = False
        self.loaded_legacy_realizable_floor = False
        self._actor_kernel = None

    def _build_networks(self) -> None:
        HumanIQLGroundedPolicy._build_networks(self)
        config = self.config
        self.recovery_encoder = ObservationEncoder(
            config.obs_dims,
            config.mechanism_obs_dims,
            config.encoder_dim,
        ).to(self.device)
        self.recovery_actor = RecoveryActorHead(
            int(self.recovery_encoder.output_dim), config.hidden_dim, config.action_dims
        ).to(self.device)
        self.action_likeness = HumanActionProximity(
            ActionProximityConfig(
                observation_dim=config.obs_dims,
                action_dim=config.action_dims,
                mechanism_dim=config.mechanism_obs_dims,
                sigma=config.action_likeness_sigma,
            )
        ).to(self.device)
        self.networks.update(
            {
                "recovery_encoder": self.recovery_encoder,
                "recovery_actor": self.recovery_actor,
                "action_likeness": self.action_likeness,
            }
        )
        self._set_sac_full_bounds()
        # Teacher starts from zero, then is initialized once after demo IQL.
        for name, encoder in (
            ("critic_1", self.encoder_critic),
            ("critic_2", self.encoder_critic),
            ("critic_target_1", self.encoder_target),
            ("critic_target_2", self.encoder_target),
        ):
            critic = ConstrainedCritic(self, encoder, getattr(self, name)).to(
                self.device
            )
            setattr(self, name, critic)
            self.networks[name] = critic
        for name in ("encoder_critic", "encoder_target"):
            encoder = RawObservation().to(self.device)
            setattr(self, name, encoder)
            self.networks[name] = encoder
        for online, target in (
            (self.critic_1, self.critic_target_1),
            (self.critic_2, self.critic_target_2),
        ):
            target.load_state_dict(online.state_dict())
            target.requires_grad_(False)
            target.eval()
        self.actor.output_activation = "algebraic"

    def _set_sac_full_bounds(self) -> None:
        """Set the inside-branch and factual Bellman bounds to [-1, 1]."""

        for critic in (
            self.critic_1,
            self.critic_2,
            self.critic_target_1,
            self.critic_target_2,
        ):
            critic.lower_bound.fill_(-1.0)

    def _build_optimizers(self) -> None:
        HumanIQLGroundedPolicy._build_optimizers(self)
        self._configure_actor_optimizer()
        self.critic_optim = torch.optim.Adam(
            [
                parameter
                for critic in (self.critic_1, self.critic_2)
                for parameter in critic.parameters()
                if parameter.requires_grad
            ],
            lr=self.config.critic_lr,
        )
        self.optimizers["critic_optimizer"] = self.critic_optim
        self.recovery_actor_optim = torch.optim.Adam(
            [
                *self.recovery_encoder.parameters(),
                *self.recovery_actor.parameters(),
            ],
            lr=self.config.recovery_actor_lr,
        )
        self.optimizers["recovery_actor_optimizer"] = self.recovery_actor_optim
        self.action_likeness_optim = torch.optim.AdamW(
            self.action_likeness.parameters(),
            lr=self.config.action_likeness_lr,
            weight_decay=self.config.action_likeness_weight_decay,
            foreach=False,
        )
        self.optimizers["action_likeness_optimizer"] = self.action_likeness_optim

    def finalize_human_iql(self) -> None:
        if self.human_iql_initialized:
            raise RuntimeError("demo critic teacher may only be initialized once")
        for critic, teacher in (
            (self.critic_1, self.human_critic_1),
            (self.critic_2, self.human_critic_2),
        ):
            critic.initialize_teacher(self.human_critic_encoder, teacher)
        for online, target in (
            (self.critic_1, self.critic_target_1),
            (self.critic_2, self.critic_target_2),
        ):
            target.load_state_dict(online.state_dict())
            target.requires_grad_(False)
            target.eval()
        self.human_iql_initialized = True
        self._build_optimizers()

    def recovery_actions(self, observation: torch.Tensor) -> torch.Tensor:
        return self.recovery_actor(self.recovery_encoder(observation, detach=False))

    def _configure_actor_optimizer(self) -> None:
        """Config wins over saved LR/backend flags; preserve Adam moments."""
        fused = self.config.fused_actor_adam and native_cuda(self.device)
        for group in self.actor_optim.param_groups:
            group.update(lr=self.config.actor_lr, fused=fused,
                         foreach=False if fused else None, capturable=False)
        for state in self.actor_optim.state.values():
            if isinstance(state.get('step'), torch.Tensor):
                state['step'] = state['step'].to(self.device if fused else 'cpu')

    def _human_floor(
        self, human_batch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from .unrecoverable import successful_only
        human_batch = successful_only(human_batch)
        if self.config.suffix_only_reference_floor:
            zero = torch.zeros((), device=self.device)
            if human_batch is None:
                return zero, {"human_floor_loss": zero, "human_floor_shortfall": zero,
                              "human_floor_violation_fraction": zero}
            obs = human_batch["observation"].to(self.device).float()
            views = int(human_batch.get("view_count", 1))
            if views < 1 or len(obs) % views:
                raise ValueError("invalid suffix floor view count")
            with torch.no_grad():
                ref = self.recovery_actions(obs)
                target = self.human_critic_min(obs, ref)
                target = target.reshape(-1, views, self.config.horizons).mean(1)
                target = target.repeat_interleave(views, 0)
            shortfalls = [
                (target - critic.reference_value(obs)).relu()
                for critic in (self.critic_1, self.critic_2)
            ]
            shortfall = torch.stack(shortfalls)
            loss = shortfall.square().mean()
            return loss, {"human_floor_loss": loss.detach(),
                          "human_floor_shortfall": shortfall.detach().mean(),
                          "human_floor_violation_fraction": (shortfall.detach() > 0).float().mean()}
        if not self.config.hard_iql_reference_floor:
            return self._human_floor_impl(human_batch)
        with torch.no_grad():
            _, metrics = self._human_floor_impl(human_batch)
        metrics.pop("human_floor_loss", None)
        return torch.zeros((), device=self.device), metrics

    def _human_floor_impl(
        self,
        human_batch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Preserve the realizable moving-IQL action in both SAC critics."""

        zero = torch.zeros((), device=self.device)
        if human_batch is None:
            return zero, {
                "human_floor_loss": zero,
                "human_floor_shortfall": zero,
                "human_floor_violation_fraction": zero,
                "human_floor_tolerance_violation_fraction": zero,
            }
        obs = cast(torch.Tensor, human_batch["observation"]).to(self.device).float()
        view_count = int(human_batch.get("view_count", 1))
        with torch.no_grad():
            iql_action = self.recovery_actions(obs)
            floor_target = self.human_critic_min(obs, iql_action)
            human_value = self.human_value(self.human_value_encoder(obs, detach=False))
            if view_count > 1 and not self.config.hard_iql_reference_floor:
                floor_target = (
                    floor_target.reshape(-1, view_count, self.config.horizons)
                    .mean(dim=1, keepdim=True)
                    .expand(-1, view_count, -1)
                    .reshape(-1, self.config.horizons)
                )
                human_value = (
                    human_value.reshape(-1, view_count, self.config.horizons)
                    .mean(dim=1, keepdim=True)
                    .expand(-1, view_count, -1)
                    .reshape(-1, self.config.horizons)
                )
        feature = self.encoder_critic(obs, detach=False)
        q1 = self.critic_1(feature, iql_action)
        q2 = self.critic_2(feature, iql_action)
        shortfall1 = torch.relu(floor_target - q1)
        shortfall2 = torch.relu(floor_target - q2)
        loss = zero if self.config.hard_iql_reference_floor else 0.5 * (
            F.smooth_l1_loss(
                shortfall1,
                torch.zeros_like(shortfall1),
                beta=self.config.human_floor_beta,
            )
            + F.smooth_l1_loss(
                shortfall2,
                torch.zeros_like(shortfall2),
                beta=self.config.human_floor_beta,
            )
        )
        q_min = torch.minimum(q1, q2)
        shortfall = torch.relu(floor_target - q_min)
        tolerance = float(self.config.human_floor_violation_tolerance)
        violating = shortfall > tolerance
        violating_float = violating.float()
        metrics = {
            "human_floor_loss": loss.detach(),
            "human_floor_shortfall": shortfall.mean().detach(),
            "human_floor_violation_fraction": (shortfall > 0).float().mean().detach(),
            "human_floor_tolerance_violation_fraction": violating_float.mean().detach(),
        }
        for horizon in MONITOR_HORIZONS:
            if horizon <= self.config.horizons:
                index = horizon - 1
                metrics[f"human_v_h{horizon}"] = human_value[:, index].mean().detach()
                metrics[f"human_floor_target_h{horizon}"] = (
                    floor_target[:, index].mean().detach()
                )
                metrics[f"human_q_h{horizon}"] = q_min[:, index].mean().detach()
                metrics[f"human_shortfall_h{horizon}"] = (
                    shortfall[:, index].mean().detach()
                )
        return loss, metrics

    def update_recovery_iql(
        self,
        human_batch: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.device)
        if human_batch is None:
            return {
                "recovery_iql_actor_loss": zero,
                "recovery_iql_advantage": zero,
                "recovery_iql_weight": zero,
            }
        obs = cast(torch.Tensor, human_batch["observation"]).to(self.device).float()
        logged_action = (
            cast(torch.Tensor, human_batch["action"]).to(self.device).float()
        )
        view_count = int(human_batch.get("view_count", 1))
        with torch.no_grad():
            q = self.human_critic_min(obs, logged_action)
            value = self.human_value(self.human_value_encoder(obs, detach=False))
            if view_count > 1:
                q = (
                    q.reshape(-1, view_count, self.config.horizons)
                    .mean(dim=1, keepdim=True)
                    .expand(-1, view_count, -1)
                    .reshape(-1, self.config.horizons)
                )
                value = (
                    value.reshape(-1, view_count, self.config.horizons)
                    .mean(dim=1, keepdim=True)
                    .expand(-1, view_count, -1)
                    .reshape(-1, self.config.horizons)
                )
            if self.config.actor_horizons is None:
                advantage = (q - value).mean(dim=1)
            else:
                indices = torch.as_tensor(
                    [h - 1 for h in self.config.actor_horizons], device=self.device
                )
                advantage = (q[:, indices] - value[:, indices]).mean(dim=1)
            weight = torch.exp(self.config.recovery_actor_beta * advantage).clamp_max(
                self.config.recovery_actor_max_weight
            )
        feature = self.recovery_encoder(obs, detach=False)
        predicted_action = self.recovery_actor(feature)
        action_error = (
            (predicted_action - logged_action.clamp(-1.0, 1.0)).square().mean(dim=1)
        )
        per_sample = action_error
        grounded_loss = (weight * per_sample).sum() / weight.sum().clamp_min(1.0)
        consistency = zero
        augmentation_std = zero
        if view_count > 1:
            grouped = predicted_action.reshape(-1, view_count, self.config.action_dims)
            consistency = (grouped - grouped.mean(dim=1, keepdim=True)).square().mean()
            augmentation_std = grouped.std(dim=1, correction=0).mean()
        loss = grounded_loss + self.config.recovery_consistency_weight * consistency
        self.recovery_actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*self.recovery_encoder.parameters(), *self.recovery_actor.parameters()],
            10.0,
        )
        self.recovery_actor_optim.step()
        return {
            "recovery_iql_actor_loss": loss.detach(),
            "recovery_iql_advantage": advantage.mean().detach(),
            "recovery_iql_weight": weight.mean().detach(),
            "recovery_iql_action_error": action_error.mean().detach(),
            "recovery_iql_consistency": consistency.detach(),
            "recovery_iql_grounded_loss": grounded_loss.detach(),
            "recovery_action_augmentation_std": augmentation_std.detach(),
        }

    def _update_moving_human_iql(
        self,
        human_batch: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        metrics = HumanIQLGroundedPolicy._update_moving_human_iql(self, human_batch)
        from .unrecoverable import successful_only
        metrics.update(self.update_recovery_iql(successful_only(human_batch)))
        return metrics

    def _replay_reference_floor(
        self, replay_batch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.config.hard_iql_reference_floor:
            return self._replay_reference_floor_impl(replay_batch)
        with torch.no_grad():
            _, metrics = self._replay_reference_floor_impl(replay_batch)
        metrics.pop("replay_reference_floor_loss", None)
        return torch.zeros((), device=self.device), metrics

    def _replay_reference_floor_impl(
        self,
        replay_batch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Ground the IQL proposal at every state sampled from replay.

        IQL itself is still updated only from D_human_suffix.  Its detached
        proposal and critic estimate are used as a generalizing floor over the
        full SAC replay distribution:

            Q_SAC(s, pi_IQL(s)) >= Q_IQL(s, pi_IQL(s)).

        Q_IQL and pi_IQL are detached. Only the SAC critic and its encoder
        receive gradients. Multiple observation augmentations are averaged
        into one constraint per physical replay state.
        """

        zero = torch.zeros((), device=self.device)
        if replay_batch is None:
            return zero, {
                "replay_reference_floor_loss": zero,
                "replay_reference_applied_fraction": zero,
                "replay_reference_shortfall": zero,
                "replay_reference_violation_fraction": zero,
                "replay_reference_target_q": zero,
                "replay_reference_sac_q": zero,
                "replay_reference_pairs": zero,
            }

        observation = (
            cast(torch.Tensor, replay_batch["observation"]).to(self.device).float()
        )
        view_count = int(replay_batch.get("view_count", 1))

        if view_count < 1 or len(observation) % view_count:
            raise ValueError("invalid grouped replay-reference view count")

        with torch.no_grad():
            iql_action = self.recovery_actions(observation)
            target = self.human_critic_min(observation, iql_action)

        feature = self.encoder_critic(observation, detach=False)
        q1 = self.critic_1(feature, iql_action)
        q2 = self.critic_2(feature, iql_action)

        if view_count > 1:
            physical_count = len(observation) // view_count

            def mean_views(value: torch.Tensor) -> torch.Tensor:
                return value.reshape(physical_count, view_count, *value.shape[1:]).mean(
                    dim=1
                )

            target = mean_views(target)
            q1 = mean_views(q1)
            q2 = mean_views(q2)

        shortfall1 = torch.relu(target - q1)
        shortfall2 = torch.relu(target - q2)
        loss = zero if self.config.hard_iql_reference_floor else 0.5 * (
            F.smooth_l1_loss(
                shortfall1,
                torch.zeros_like(shortfall1),
                beta=self.config.human_floor_beta,
            )
            + F.smooth_l1_loss(
                shortfall2,
                torch.zeros_like(shortfall2),
                beta=self.config.human_floor_beta,
            )
        )
        q_min = torch.minimum(q1, q2)
        shortfall = torch.relu(target - q_min)
        tolerance = float(self.config.human_floor_violation_tolerance)

        return loss, {
            "replay_reference_floor_loss": loss.detach(),
            "replay_reference_applied_fraction": torch.ones((), device=self.device),
            "replay_reference_shortfall": shortfall.mean().detach(),
            "replay_reference_violation_fraction": (shortfall > tolerance)
            .float()
            .mean()
            .detach(),
            "replay_reference_target_q": target.mean().detach(),
            "replay_reference_sac_q": q_min.mean().detach(),
            "replay_reference_pairs": torch.as_tensor(
                float(shortfall.numel()), device=self.device
            ),
        }

    def export(self) -> DictMessage:
        state = HumanIQLGroundedPolicy.export(self)
        state["unrecoverable_value_supervision"] = "iql_failure_suffix_v2"
        state["algorithm"] = (
            "limit_action_v15_suffix_base_floor"
            if self.config.suffix_only_reference_floor else
            "limit_action_v14_annotation_only"
            if self.config.hard_iql_reference_floor
            else "limit_action_v10_constrained_q"
        )
        state["recovery_initialized"] = self.recovery_initialized
        state["action_fence_reference_ratio"] = (
            self.config.action_likeness_reference_ratio
        )
        state["actor_output_activation"] = self.actor.output_activation
        return state

    def load(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") not in (
            "limit_action_v10_constrained_q", "limit_action_v11_hard_iql_reference",
            "limit_action_v12_failure_state", "limit_action_v13_failure_vector",
            "limit_action_v14_annotation_only", "limit_action_v15_suffix_base_floor"
        ):
            raise ValueError(
                "incompatible constrained-Q checkpoint; start a new experiment "
                "with the existing compatible raw demos, not a legacy learner checkpoint"
            )
        if state.get("actor_output_activation") != "algebraic":
            raise ValueError("constrained-Q checkpoint requires algebraic actor output")
        ratio = state.get("action_fence_reference_ratio")
        if not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
            raise ValueError("invalid checkpoint fence reference ratio")
        for name in (*self.networks, "training_state"):
            if not isinstance(state.get(name), dict):
                raise ValueError(f"constrained-Q checkpoint missing {name!r}")
        compatible = dict(state)
        compatible["algorithm"] = "human_iql_grounded_actor_critic_v2_independent_iql"
        HumanIQLGroundedPolicy.load(self, compatible)
        self._configure_actor_optimizer()
        self._actor_kernel = None
        self.config.suffix_only_reference_floor = state["algorithm"] == "limit_action_v15_suffix_base_floor"
        # Never reinterpret a resumed v10 base head as a nonnegative v11 head.
        self.config.hard_iql_reference_floor = (
            state["algorithm"] in (
                "limit_action_v11_hard_iql_reference", "limit_action_v12_failure_state",
                "limit_action_v13_failure_vector", "limit_action_v14_annotation_only",
            )
        )
        self.config.action_likeness_reference_ratio = ratio
        self.recovery_initialized = bool(state.get("recovery_initialized", False))
        self.actor.output_activation = "algebraic"
        if bool(self.action_likeness.initialized.item()):
            self.action_likeness.requires_grad_(False)
            self.action_likeness.eval()
