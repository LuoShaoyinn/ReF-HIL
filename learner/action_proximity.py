from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

ACTION_LIKENESS_SIGMA = 1.0 / 6.0
ACTION_LIKENESS_QUERY_STD = 0.1


def gaussian_density_peak(
    action_dim: int, *, sigma: float = ACTION_LIKENESS_SIGMA
) -> float:
    """Peak of the normalized N(0, sigma^2 I) density on R^D."""

    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return (2.0 * math.pi * float(sigma) ** 2) ** (-0.5 * action_dim)


def proximity_threshold_from_radius(
    radius: float,
    *,
    sigma: float = ACTION_LIKENESS_SIGMA,
    action_dim: int = 4,
) -> float:
    """Convert an L2 radius to a normalized Gaussian-density threshold."""

    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return gaussian_density_peak(action_dim, sigma=sigma) * math.exp(
        -(float(radius) ** 2) / (2.0 * float(sigma) ** 2)
    )


def gaussian_density_threshold_for_mass(
    action_dim: int,
    *,
    retained_mass: float,
    sigma: float = ACTION_LIKENESS_SIGMA,
) -> tuple[float, float]:
    """Density level whose R^D Gaussian superlevel set retains given mass."""

    if not 0.0 < retained_mass < 1.0:
        raise ValueError("retained_mass must be in (0,1)")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    shape = torch.tensor(0.5 * action_dim, dtype=torch.float64)
    lower = 0.0
    upper = float(action_dim)
    while float(torch.special.gammainc(shape, torch.tensor(0.5 * upper))) < retained_mass:
        upper *= 2.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        cdf = float(torch.special.gammainc(shape, torch.tensor(0.5 * middle)))
        if cdf < retained_mass:
            lower = middle
        else:
            upper = middle
    radius = float(sigma) * math.sqrt(0.5 * (lower + upper))
    return proximity_threshold_from_radius(
        radius, sigma=sigma, action_dim=action_dim
    ), radius


def _standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def sample_truncated_action_normal(
    center: torch.Tensor, *, std: float = ACTION_LIKENESS_QUERY_STD
) -> torch.Tensor:
    """Sample N(center, std^2 I) conditioned on the action cube."""

    if std <= 0.0:
        raise ValueError("std must be positive")
    lower = _standard_normal_cdf((-1.0 - center) / std)
    upper = _standard_normal_cdf((1.0 - center) / std)
    probability = lower + (upper - lower) * torch.rand_like(center)
    probability = probability.clamp(1e-7, 1.0 - 1e-7)
    standardized = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    return (center + std * standardized).clamp(-1.0, 1.0)


def action_query_proposal_density(
    action: torch.Tensor,
    center: torch.Tensor,
    *,
    local_std: float = ACTION_LIKENESS_QUERY_STD,
) -> torch.Tensor:
    """Density of 0.5 Uniform(cube) + 0.5 truncated local Gaussian."""

    if local_std <= 0.0:
        raise ValueError("local_std must be positive")
    action_dim = action.shape[-1]
    uniform_density = 2.0 ** (-action_dim)
    standardized = (action - center) / local_std
    lower = _standard_normal_cdf((-1.0 - center) / local_std)
    upper = _standard_normal_cdf((1.0 - center) / local_std)
    normalizer = (upper - lower).clamp_min(1e-12)
    component_density = (
        torch.exp(-0.5 * standardized.square())
        / (math.sqrt(2.0 * math.pi) * local_std * normalizer)
    )
    local_density = component_density.prod(dim=-1)
    return 0.5 * uniform_density + 0.5 * local_density


@dataclass(frozen=True)
class ActionProximityConfig:
    observation_dim: int
    action_dim: int = 4
    mechanism_dim: int = 10
    mechanism_hidden_dim: int = 64
    image_hidden_dim: int = 256
    state_dim: int = 256
    interaction_dim: int = 128
    hidden_dim: int = 256
    sigma: float = ACTION_LIKENESS_SIGMA


class HumanActionProximity(nn.Module):
    """H(s, a) estimating proximity to demonstrated human actions."""

    def __init__(self, config: ActionProximityConfig) -> None:
        super().__init__()
        self.config = config
        if config.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        image_dim = config.observation_dim - config.mechanism_dim
        if image_dim < 0:
            raise ValueError("observation_dim must be at least mechanism_dim")
        self.mechanism_encoder = nn.Sequential(
            nn.Linear(config.mechanism_dim, config.mechanism_hidden_dim),
            nn.LayerNorm(config.mechanism_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.mechanism_hidden_dim, config.mechanism_hidden_dim),
            nn.SiLU(),
        )
        self.image_encoder = (
            nn.Sequential(
                nn.Linear(image_dim, config.image_hidden_dim),
                nn.LayerNorm(config.image_hidden_dim),
                nn.SiLU(),
            )
            if image_dim
            else None
        )
        state_input = config.mechanism_hidden_dim + (
            config.image_hidden_dim if image_dim else 0
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input, config.state_dim),
            nn.LayerNorm(config.state_dim),
            nn.SiLU(),
        )
        self.interaction = nn.Bilinear(
            config.state_dim, config.action_dim, config.interaction_dim
        )
        self.head = nn.Sequential(
            nn.Linear(
                config.state_dim + config.action_dim + config.interaction_dim,
                config.hidden_dim,
            ),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.register_buffer("observation_mean", torch.zeros(config.observation_dim))
        self.register_buffer("observation_std", torch.ones(config.observation_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def set_normalization(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> None:
        self.observation_mean.copy_(observation.mean(dim=0))
        self.observation_std.copy_(observation.std(dim=0).clamp_min(1e-4))
        self.action_mean.copy_(action.mean(dim=0))
        self.action_std.copy_(action.std(dim=0).clamp_min(1e-4))

    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        observation = (observation - self.observation_mean) / self.observation_std
        mechanism = self.mechanism_encoder(
            observation[..., : self.config.mechanism_dim]
        )
        if self.image_encoder is None:
            state_input = mechanism
        else:
            image = self.image_encoder(
                observation[..., self.config.mechanism_dim :]
            )
            state_input = torch.cat((mechanism, image), dim=-1)
        return self.state_encoder(state_input)

    def logits_from_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        action = (action - self.action_mean) / self.action_std
        interaction = self.interaction(state, action)
        return self.head(torch.cat((state, action, interaction), dim=-1)).squeeze(-1)

    def logits(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.logits_from_state(self.encode_observation(observation), action)

    @property
    def density_peak(self) -> float:
        return gaussian_density_peak(
            self.config.action_dim, sigma=self.config.sigma
        )

    def density_ratio(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Density divided by the fixed Gaussian peak, in (0, 1)."""

        return torch.sigmoid(self.logits(observation, action))

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Normalized action density on R^D, evaluated inside the action cube."""

        return self.density_peak * self.density_ratio(observation, action)

    def density_from_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.density_peak * torch.sigmoid(
            self.logits_from_state(state, action)
        )

    def finalize(self) -> None:
        self.initialized.fill_(True)
        self.requires_grad_(False)
        self.eval()

    def enable_online_training(self) -> None:
        """Continue fitting H on human actions in the learner process."""

        if not bool(self.initialized.item()):
            raise RuntimeError("action proximity must be pretrained first")
        self.requires_grad_(True)
        self.train()


class RecoveryActorHead(nn.Module):
    """Deterministic policy with every action dimension in [-1, 1]."""

    def __init__(self, feature_dim: int, hidden_dim: int, action_dims: int = 4) -> None:
        super().__init__()
        if action_dims < 1:
            raise ValueError("action_dims must be positive")
        self.action_dims = int(action_dims)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, self.action_dims)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(feature)
        return torch.tanh(self.output(hidden))


def proximity_target(
    action: torch.Tensor,
    human_action: torch.Tensor,
    *,
    sigma: float = ACTION_LIKENESS_SIGMA,
) -> torch.Tensor:
    """Normalized Gaussian-density target on R^D."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    distance_sq = (action - human_action).square().sum(dim=-1)
    peak = gaussian_density_peak(action.shape[-1], sigma=sigma)
    return peak * torch.exp(-distance_sq / (2.0 * float(sigma) ** 2))
