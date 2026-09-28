from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from shared.actor_network import DeterministicActorHead


def horizon_lower_bound(
    horizons: int,
    gamma: float,
    minimum_step_reward: float,
) -> torch.Tensor:
    h = torch.arange(1, horizons + 1, dtype=torch.float32)
    if math.isclose(gamma, 1.0):
        return minimum_step_reward * h
    return minimum_step_reward * (1.0 - gamma**h) / (1.0 - gamma)


def horizon_extension_violation(
    value: torch.Tensor,
    step_penalty: float,
) -> torch.Tensor:
    """Violation of f[h] >= f[h-1] - step_penalty on raw heads."""

    if value.ndim != 2:
        raise ValueError("horizon value must have shape [batch, horizons]")
    if value.shape[1] <= 1:
        return value.new_zeros((len(value), 0))
    return torch.relu(value[:, :-1] - float(step_penalty) - value[:, 1:])


def horizon_ranking_loss(
    value: torch.Tensor,
    step_penalty: float,
) -> torch.Tensor:
    violation = horizon_extension_violation(value, step_penalty)
    if violation.numel() == 0:
        return value.new_zeros(())
    return violation.square().mean()


def expectile_loss(
    difference: torch.Tensor,
    expectile: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = torch.where(difference > 0.0, expectile, 1.0 - expectile)
    loss = weight * difference.square()
    if mask is None:
        return loss.mean()
    float_mask = mask.to(dtype=loss.dtype)
    return (loss * float_mask).sum() / float_mask.sum().clamp_min(1.0)


def build_vector_bellman_target(
    reward: torch.Tensor,
    success: torch.Tensor,
    timeout: torch.Tensor,
    timeout_next_valid: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
    lower_bound: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift h to h-1, with V_0=0 and explicit timeout-next support."""

    batch, horizons = next_value.shape
    target = reward[:, None].expand(batch, horizons).clone()
    valid = torch.ones_like(target, dtype=torch.bool)
    can_bootstrap = ~success & (~timeout | timeout_next_valid)
    missing_timeout = timeout & ~timeout_next_valid
    if horizons > 1:
        target[:, 1:] += (
            gamma
            * next_value[:, :-1]
            * can_bootstrap[:, None].to(dtype=next_value.dtype)
        )
        valid[:, 1:] &= ~missing_timeout[:, None]
    # A successful terminal may carry graded task quality. Preserve the
    # recorded reward on every horizon instead of silently rewriting it to 1.
    target = torch.where(success[:, None], reward[:, None], target)
    target = torch.maximum(target, lower_bound[None, :]).clamp_max(1.0)
    return target, valid


class LowRankActionGatedLayer(nn.Module):
    """Each raw action scalar gates a low-rank state transformation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.rank = int(rank)
        self.base = nn.Linear(input_dim, output_dim)
        self.state_factors = nn.Linear(input_dim, action_dim * rank, bias=False)
        self.action_projections = nn.Parameter(
            torch.empty(action_dim, rank, output_dim)
        )
        nn.init.xavier_uniform_(self.action_projections)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        factors = self.state_factors(state).reshape(
            len(state), self.action_dim, self.rank
        )
        interaction = torch.einsum(
            "bar,ba,aro->bo",
            factors,
            action,
            self.action_projections,
        )
        return self.base(state) + interaction


class BoundedVectorHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lower_bound: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("lower_bound", lower_bound.detach().clone())
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(lower_bound)),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(self.net(feature))
        return torch.where(unit >= 0.0, unit, unit * (-self.lower_bound))

class BilinearVectorCritic(nn.Module):
    """Bounded Q(s,a)[1..H] with action gating in every state block."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        lower_bound: torch.Tensor,
        rank: int,
    ) -> None:
        super().__init__()
        self.register_buffer("lower_bound", lower_bound.detach().clone())
        self.input_gate = LowRankActionGatedLayer(
            feature_dim, hidden_dim, action_dim, rank
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.residual_gate_1 = LowRankActionGatedLayer(
            hidden_dim, hidden_dim, action_dim, rank
        )
        self.residual_norm_1 = nn.LayerNorm(hidden_dim)
        self.residual_gate_2 = LowRankActionGatedLayer(
            hidden_dim, hidden_dim, action_dim, rank
        )
        self.residual_norm_2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, len(lower_bound))

    def forward(self, feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input_norm(self.input_gate(feature, action)))
        hidden = hidden + F.silu(
            self.residual_norm_1(self.residual_gate_1(hidden, action))
        )
        hidden = hidden + F.silu(
            self.residual_norm_2(self.residual_gate_2(hidden, action))
        )
        unit = torch.tanh(self.output(hidden))
        return torch.where(unit >= 0.0, unit, unit * (-self.lower_bound))
