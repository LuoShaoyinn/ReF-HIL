"""Reference-centered SAC critic with an analytical outside-fence field.

The outside branch is an optimization surrogate, not an environmental return.
The policy reference is weak/non-owning: IQL and H are not critic parameters.
"""

from __future__ import annotations

import copy
import weakref
from typing import TYPE_CHECKING

import torch
from torch import nn

from .iql_modules import BilinearVectorCritic, BoundedVectorHead
from .observation_encoder import ObservationEncoder

if TYPE_CHECKING:
    from .policy import LimitActionPolicy


class RawObservation(nn.Module):
    """The two critic branches now own their separate observation encoders."""

    def forward(self, observation: torch.Tensor, detach: bool = False) -> torch.Tensor:
        return observation.detach() if detach else observation


class ConstrainedCritic(nn.Module):
    def __init__(
        self,
        policy: LimitActionPolicy,
        encoder: nn.Module,
        teacher: BilinearVectorCritic,
    ) -> None:
        super().__init__()
        self._policy = weakref.ref(policy)
        config = policy.config
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher = copy.deepcopy(teacher)
        self.register_buffer("lower_bound", teacher.lower_bound.clone())
        self.base_encoder = ObservationEncoder(
            config.obs_dims, config.mechanism_obs_dims, config.encoder_dim
        )
        self.adv_encoder = ObservationEncoder(
            config.obs_dims, config.mechanism_obs_dims, config.encoder_dim
        )
        self.base = BoundedVectorHead(
            self.base_encoder.output_dim, config.hidden_dim, self.lower_bound
        )
        self.adv = BilinearVectorCritic(
            self.adv_encoder.output_dim,
            config.action_dims,
            config.hidden_dim,
            self.lower_bound,
            config.bilinear_rank,
        )
        self.reset_residual()
        self.requires_grad_(True)

    def reset_residual(self) -> None:
        nn.init.zeros_(self.base.net[-1].weight)
        policy = self._policy()
        nn.init.constant_(
            self.base.net[-1].bias,
            -4.0 if policy.config.hard_iql_reference_floor
            and not policy.config.suffix_only_reference_floor else 0.0,
        )
        nn.init.zeros_(self.adv.output.weight)
        nn.init.zeros_(self.adv.output.bias)

    def initialize_teacher(self, encoder: nn.Module, critic: nn.Module) -> None:
        self.teacher_encoder.load_state_dict(encoder.state_dict())
        self.teacher.load_state_dict(critic.state_dict())
        self.teacher.lower_bound.fill_(-1.0)
        self.reset_residual()

    def requires_grad_(self, requires_grad: bool = True) -> ConstrainedCritic:
        super().requires_grad_(requires_grad)
        # Parent SAC toggles critic gradients around each actor update.
        self.teacher.requires_grad_(False)
        self.teacher_encoder.requires_grad_(False)
        return self

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        *,
        gate_observation: torch.Tensor | None = None,
        rank_only: bool = False,
    ) -> torch.Tensor:
        policy = self._policy()
        if policy is None:
            raise RuntimeError("critic owner no longer exists")
        gate = observation if gate_observation is None else gate_observation
        with torch.no_grad():
            reference = policy.recovery_actions(gate)
            rejected = policy.rejected_actions(gate, action, reference)
        teacher_feature = self.teacher_encoder(observation)
        teacher_reference = self.teacher(teacher_feature, reference)
        if policy.config.suffix_only_reference_floor:
            baseline = self.reference_value(observation)
        elif policy.config.hard_iql_reference_floor:
            with torch.no_grad():
                value = policy.human_critic_min(gate, reference)
            # Effective B is nonnegative and bounded by available headroom.
            # Unlike ReLU it has no dead negative-input region.
            bonus = (1.0 - value).clamp_min(0) * torch.sigmoid(
                self.base.net(self.base_encoder(observation))
            )
            baseline = value + bonus
        else:
            baseline = (
                teacher_reference + self.base(self.base_encoder(observation))
            ).clamp(-1, 1)
        if rank_only:
            baseline = baseline.detach()
        feature = self.adv_encoder(observation)
        residual = (
            self.teacher(teacher_feature, action)
            - teacher_reference
            + self.adv(feature, action)
            - self.adv(feature, reference)
        )
        inside = baseline + residual
        if not policy.config.suffix_only_reference_floor:
            inside = inside.clamp(-1, 1)
        outside = (
            baseline
            - policy.config.stay_step_penalty
            - (action - reference).square().sum(-1, keepdim=True)
        )
        # Clipping outside to -1 would erase its restoring action gradient.
        return torch.where(rejected[:, None], outside, inside)

    def reference_value(self, observation):
        """Unconstrained B; no IQL offset, A graph, or positivity activation."""
        return self.base.net(self.base_encoder(observation))

    def ranking_advantage(self, observation, action, reference):
        """Unclipped centered advantage; differentiate both residual evaluations.

        The reference action is a detached label, not a detached network
        evaluation. B and the frozen teacher parameters never receive this loss.
        """
        action, reference = action.detach(), reference.detach()
        feature = self.adv_encoder(observation)
        teacher_feature = self.teacher_encoder(observation)
        value = self.teacher(teacher_feature, action) + self.adv(feature, action)
        anchor = self.teacher(teacher_feature, reference) + self.adv(
            feature, reference
        )
        return value - anchor


def factual_td_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    rejected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain factual all-horizon TD only for fence-accepted recorded pairs."""
    mask = valid & ~rejected[:, None]
    denominator = mask.sum().clamp_min(1)
    smooth = sum(
        torch.nn.functional.smooth_l1_loss(q, target, beta=0.05, reduction="none")
        for q in (q1, q2)
    )
    mse = sum((q - target).square() for q in (q1, q2))
    return (smooth * mask).sum() / denominator, (mse * mask).sum() / denominator, mask
