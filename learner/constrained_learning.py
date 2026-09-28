"""SAC losses for the reference-centered constrained critic.

Independent IQL/H training, replay and checkpoint ownership stay in policy.py.
This mixin has no runtime action substitution and no separate imitation loss.
"""

from __future__ import annotations

import torch

from .constrained_critic import factual_td_loss
from .iql_modules import build_vector_bellman_target, horizon_ranking_loss
from .actor_optimization import native_cuda, prepare_actor_cache, cached_actor_objective
from .unrecoverable import failure_vector, state_supervision


class ConstrainedLearning:
    @torch.no_grad()
    def rejected_actions(self, observation, action, reference=None):
        if reference is None:
            reference = self.recovery_actions(observation)
        return self.action_likeness(observation, action) < self._fence_threshold(
            observation, reference
        )

    @torch.no_grad()
    def _certified_proposal_continuation(
        self,
        next_observation,
        certified_suffix_next,
        view_count,
        gate_next_observation=None,
    ):
        # Retain the caller's interface; continuation applies on all replay.
        del certified_suffix_next
        if view_count < 1 or len(next_observation) % view_count:
            raise ValueError("invalid grouped continuation view count")
        gate = (
            next_observation if gate_next_observation is None else gate_next_observation
        )
        sac = self.actor_actions(next_observation)
        reference = self.recovery_actions(gate)

        def target_min(action):
            return torch.minimum(
                *(
                    critic(next_observation, action, gate_observation=gate)
                    for critic in (self.critic_target_1, self.critic_target_2)
                )
            )

        proposal_q = target_min(sac)
        reference_q = target_min(reference)
        iql_q = self.human_critic_min(next_observation, reference)
        # Explicitly include B from the SAME target critics. Outside Q never
        # wins against this candidate, irrespective of IQL/SAC level mismatch.
        continuation = torch.maximum(proposal_q, reference_q)
        if not self.config.suffix_only_reference_floor:
            continuation = torch.maximum(continuation, iql_q)
        metrics = {
            "target_rejected_fraction": self.rejected_actions(gate, sac, reference)
            .float()
            .mean(),
            "target_iql_selected_fraction": (
                (iql_q > torch.maximum(proposal_q, reference_q))
                & (not self.config.suffix_only_reference_floor)
            )
            .float()
            .mean(),
        }
        if view_count > 1:
            continuation = continuation.reshape(
                -1, view_count, self.config.horizons
            ).mean(1)
            continuation = continuation.repeat_interleave(view_count, 0)
        return continuation, metrics

    def _correction_rank_loss(self, correction_batch):
        zero = torch.zeros((), device=self.device)
        if correction_batch is None:
            return zero, {
                "correction_rank_loss": zero,
                "correction_violation_fraction": zero,
                "correction_cloud_valid_fraction": zero,
            }
        observation = correction_batch["observation"].to(self.device).float()
        action = correction_batch["autonomous_action"].to(self.device).float().detach()
        views = int(correction_batch.get("view_count", 1))
        if views < 1 or len(observation) % views:
            raise ValueError("invalid grouped correction view count")
        count = len(observation) // views
        cloud_count = 1 + self.config.correction_noise_samples
        cloud = (
            action.reshape(count, views, -1)[:, 0, None]
            .expand(-1, cloud_count, -1)
            .clone()
        )
        if cloud_count > 1:
            cloud[:, 1:] += self.config.correction_noise_std * torch.randn_like(
                cloud[:, 1:]
            )
        cloud.clamp_(-1, 1)
        obs = (
            observation.reshape(count, views, -1)[:, None]
            .expand(-1, cloud_count, -1, -1)
            .reshape(-1, observation.shape[-1])
        )
        act = (
            cloud[:, :, None]
            .expand(-1, -1, views, -1)
            .reshape(-1, self.config.action_dims)
        )
        with torch.no_grad():
            ref = self.recovery_actions(obs)
            distance = (act - ref).norm(dim=-1)
            valid = (
                distance >= self.config.correction_reference_exclusion_radius
            )
        # Human corrections supervise the latent advantage even outside the
        # fence, where the hard override would otherwise hide violations.
        losses, violations = [], []
        for critic in (self.critic_1, self.critic_2):
            gap = critic.ranking_advantage(obs, act, ref).mean(-1)
            gap = gap + self.config.stay_step_penalty
            gap = gap.reshape(count, cloud_count, views)
            weights = valid.reshape(count, cloud_count, views).float()
            # Match the tested variant: mean horizon first, then a linear
            # hinge on each accepted observation view of the cloud action.
            denominator = weights.sum().clamp_min(1)
            losses.append((gap.relu() * weights).sum() / denominator)
            violations.append(((gap.detach() > 0) * weights).sum() / denominator)
        loss = sum(losses) / 2
        return loss, {
            "correction_rank_loss": loss.detach(),
            "correction_violation_fraction": sum(violations) / 2,
            "correction_cloud_valid_fraction": valid.float().mean(),
        }

    def _rejected_advantage_loss(self, obs, recorded_action, gate):
        # Cover both actually executed rejected pairs and current proposals.
        # Actions, IQL and H are labels here, not optimization targets.
        with torch.no_grad():
            actions = torch.cat((recorded_action, self.actor_actions(obs)), 0).detach()
            gates = torch.cat((gate, gate), 0)
            refs = self.recovery_actions(gates)
            rejected = self.rejected_actions(gates, actions, refs)
        observations = torch.cat((obs, obs), 0)
        denominator = rejected.sum().clamp_min(1)
        losses, violations = [], []
        for critic in (self.critic_1, self.critic_2):
            gap = critic.ranking_advantage(observations, actions, refs).mean(-1)
            gap = gap + self.config.stay_step_penalty
            losses.append((gap.relu() * rejected).sum() / denominator)
            violations.append(((gap.detach() > 0) * rejected).sum() / denominator)
        return sum(losses) / 2, {
            "rejected_advantage_rank_loss": (sum(losses) / 2).detach(),
            "rejected_advantage_violation_fraction": sum(violations) / 2,
            "rejected_advantage_sample_fraction": rejected.float().mean(),
        }

    def _critic_loss(
        self,
        *,
        obs,
        action,
        reward,
        success,
        timeout,
        timeout_next_valid,
        next_obs,
        human_batch,
        unrecoverable_next=None,
        unrecoverable_observation=None,
        gate_obs=None,
        gate_next_obs=None,
        view_count=1,
        ood_incoming_mask=None,
        correction_batch=None,
        certified_suffix_next=None,
    ):
        del ood_incoming_mask
        gate = obs if gate_obs is None else gate_obs
        q1, q2 = (
            critic(obs, action, gate_observation=gate)
            for critic in (self.critic_1, self.critic_2)
        )
        with torch.no_grad():
            rejected = self.rejected_actions(gate, action)
            continuation, target_metrics = self._certified_proposal_continuation(
                next_obs, certified_suffix_next, view_count, gate_next_obs
            )
            if unrecoverable_next is not None:
                failed = unrecoverable_next.to(device=obs.device, dtype=torch.bool)
                failure = failure_vector(self.config.horizons, self.config.gamma,
                                         self.config.stay_step_penalty, device=obs.device)
                continuation = torch.where(failed[:, None], failure[None, :], continuation)
                timeout_next_valid = timeout_next_valid | failed
                # Actual evidence of entering an unrecoverable state must not
                # disappear behind the action-fence TD mask.
                rejected = rejected & ~failed
            target, valid = build_vector_bellman_target(
                reward,
                success,
                timeout,
                timeout_next_valid,
                continuation,
                gamma=self.config.gamma,
                lower_bound=self.critic_1.lower_bound,
            )
        if unrecoverable_next is not None:
            # Fit the raw environmental field, not the analytical outside-Q
            # surrogate, on explicitly failed incoming transitions.
            with torch.no_grad():
                ref = self.recovery_actions(gate)
            raw_q = []
            for critic in (self.critic_1, self.critic_2):
                tf = critic.teacher_encoder(obs)
                af = critic.adv_encoder(obs)
                raw_q.append(critic.reference_value(obs)
                    + critic.teacher(tf, action) + critic.adv(af, action)
                    - critic.teacher(tf, ref) - critic.adv(af, ref))
            q1 = torch.where(unrecoverable_next[:, None], raw_q[0], q1)
            q2 = torch.where(unrecoverable_next[:, None], raw_q[1], q2)
        smooth, mse, mask = factual_td_loss(q1, q2, target, valid, rejected)
        rank = sum(
            horizon_ranking_loss(q, self.config.stay_step_penalty) for q in (q1, q2)
        )
        floor, floor_metrics = self._human_floor(human_batch)
        replay_floor, replay_metrics = obs.new_zeros(()), {}
        if not self.config.suffix_only_reference_floor:
            replay_floor, replay_metrics = self._replay_reference_floor(
                {"observation": obs, "view_count": view_count}
            )
        correction, correction_metrics = self._correction_rank_loss(correction_batch)
        reject_rank, reject_metrics = self._rejected_advantage_loss(obs, action, gate)
        failure_loss = state_supervision(self, unrecoverable_observation)
        loss = (
            smooth + failure_loss
            + self.config.td_mse_weight * mse
            + self.config.horizon_rank_weight * rank
            + self.config.correction_rank_weight * correction
            + self.config.rejected_advantage_rank_weight * reject_rank
        )
        if self.config.suffix_only_reference_floor or not self.config.hard_iql_reference_floor:
            loss = loss + self.config.human_floor_weight * floor
            loss = loss + self.config.replay_reference_floor_weight * replay_floor
        return loss, {
            "critic_td_smooth_loss": smooth.detach(),
            "unrecoverable_state_loss": failure_loss.detach(),
            "critic_td_mse_loss": mse.detach(),
            "critic_horizon_rank_loss": rank.detach(),
            "critic_valid_pairs": mask.sum().float(),
            "recorded_rejected_fraction": rejected.float().mean(),
            **target_metrics,
            **floor_metrics,
            **replay_metrics,
            **correction_metrics,
            **reject_metrics,
        }

    def _actor_loss(self, obs, gate_obs=None, view_count=1):
        if view_count < 1 or len(obs) % view_count:
            raise ValueError("invalid grouped actor view count")
        if (self.config.optimize_actor and self.config.suffix_only_reference_floor
                and not obs.requires_grad):
            gate = obs if gate_obs is None else gate_obs
            cache = prepare_actor_cache(self, obs, gate)
            if self.config.compile_actor and native_cuda(self.device):
                if self._actor_kernel is None:
                    self._actor_kernel = torch.compile(cached_actor_objective, fullgraph=True)
                    print('[ACTOR] compiling cached CUDA objective; first update includes warm-up', flush=True)
                kernel = self._actor_kernel
            else:
                kernel = cached_actor_objective
            loss, (q, action, rejected, consistency) = kernel(self, obs, view_count, cache)
            return loss, self._actor_metrics(q, action, cache[0], rejected, consistency)
        action = self.actor_actions(obs)
        q = torch.minimum(
            *(
                critic(obs, action, gate_observation=gate_obs)
                for critic in (self.critic_1, self.critic_2)
            )
        )
        loss = -q.mean()
        consistency = obs.new_zeros(())
        if view_count < 1 or len(obs) % view_count:
            raise ValueError("invalid grouped actor view count")
        if view_count > 1:
            grouped = action.reshape(-1, view_count, action.shape[-1])
            consistency = (grouped - grouped.mean(1, keepdim=True)).square().mean()
            loss = loss + self.config.actor_consistency_weight * consistency
        with torch.no_grad():
            gate = obs if gate_obs is None else gate_obs
            reference = self.recovery_actions(gate)
            rejected = self.rejected_actions(gate, action, reference)
        return loss, self._actor_metrics(q, action, reference, rejected, consistency)

    def _actor_metrics(self, q, action, reference, rejected, consistency):
        with torch.no_grad():
            distance = (action-reference).norm(dim=-1)
            rejected_distance = (distance*rejected).sum() / rejected.sum().clamp_min(1)
        metrics = {
            "actor_standard_loss": -q.detach().mean(),
            "actor_raw_q": q.detach().mean(),
            "actor_consistency_loss": consistency.detach(),
            "actor_rejected_fraction": rejected.float().mean(),
            "actor_reference_l2": distance.mean(),
            "actor_rejected_l2": rejected_distance,
            "actor_saturation_fraction": (action.detach().abs() > 0.99).float().mean(),
        }
        for horizon in (10, 20, 50, 100):
            if horizon <= self.config.horizons:
                metrics[f"actor_q_h{horizon}"] = q[:, horizon - 1].detach().mean()
        return metrics
