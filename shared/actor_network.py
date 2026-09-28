from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


ACTOR_SNAPSHOT_FORMAT = "deterministic_actor_v1"


class ObservationEncoder(nn.Module):
    """Encode proprioception and retain frozen image features verbatim."""

    def __init__(
        self,
        obs_dim: int,
        mechanism_obs_dims: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        if not 0 < mechanism_obs_dims <= obs_dim:
            raise ValueError(
                "mechanism_obs_dims must be in [1, obs_dim], got "
                f"{mechanism_obs_dims} for obs_dim={obs_dim}"
            )
        self.mechanism_obs_dims = int(mechanism_obs_dims)
        self.image_feature_dims = int(obs_dim - mechanism_obs_dims)
        self.output_dim = int(feature_dim + self.image_feature_dims)
        self.net = nn.Sequential(
            nn.Linear(mechanism_obs_dims, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        mechanism_obs = obs[..., : self.mechanism_obs_dims]
        image_features = obs[..., self.mechanism_obs_dims :]
        mechanism_feat = self.net(mechanism_obs)
        feature = (
            mechanism_feat
            if self.image_feature_dims == 0
            else torch.cat((mechanism_feat, image_features), dim=-1)
        )
        return feature.detach() if detach else feature


class DeterministicActorHead(nn.Module):
    """Deterministic actor with every action dimension in [-1, 1]."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        action_dims: int,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if action_dims < 1:
            raise ValueError("action_dims must be positive")
        self.action_dims = int(action_dims)
        self.output_activation = output_activation
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Retain this public name for existing learner checkpoints.
        self.continuous = nn.Linear(hidden_dim, self.action_dims)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        logits = self.continuous(self.backbone(feature))
        if self.output_activation == "algebraic":
            return logits / torch.sqrt(1 + logits.square())
        if self.output_activation == "tanh":
            return torch.tanh(logits)
        raise ValueError(f"unknown actor output activation: {self.output_activation!r}")


@dataclass(frozen=True, kw_only=True)
class ActorNetworkConfig:
    obs_dims: int
    mechanism_obs_dims: int
    encoder_dim: int
    hidden_dim: int
    action_dims: int

    def as_dict(self) -> dict[str, int]:
        return {
            "obs_dims": int(self.obs_dims),
            "mechanism_obs_dims": int(self.mechanism_obs_dims),
            "encoder_dim": int(self.encoder_dim),
            "hidden_dim": int(self.hidden_dim),
            "action_dims": int(self.action_dims),
        }


class ActorInferencePolicy:
    """Inference-only policy with no learner or algorithm dependency."""

    def __init__(self, *, config: ActorNetworkConfig, device: str) -> None:
        self.config = config
        self.device = torch.device(device)
        self.encoder = ObservationEncoder(
            config.obs_dims,
            config.mechanism_obs_dims,
            config.encoder_dim,
        ).to(self.device)
        self.actor = DeterministicActorHead(
            self.encoder.output_dim,
            config.hidden_dim,
            config.action_dims,
        ).to(self.device)
        self.encoder.eval()
        self.actor.eval()

    def sample_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del deterministic
        obs = observation.to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor(self.encoder(obs))
        return action.squeeze(0)

    def load(self, state: dict[str, Any]) -> None:
        if state.get("format") != ACTOR_SNAPSHOT_FORMAT:
            raise ValueError(f"incompatible actor snapshot: {state.get('format')!r}")
        if state.get("config") != self.config.as_dict():
            raise ValueError(
                "actor snapshot architecture mismatch: "
                f"snapshot={state.get('config')!r}, runtime={self.config.as_dict()!r}"
            )
        activation = state.get("output_activation", "tanh")
        if activation not in ("tanh", "algebraic"):
            raise ValueError(f"unknown actor output activation: {activation!r}")
        if any(
            state.get(key) is not None
            for key in (
                "reference_encoder",
                "reference_actor",
                "reference_max_distance",
            )
        ):
            raise ValueError("runtime reference fallback snapshots are not supported")
        self.actor.output_activation = activation
        for key, module in (("encoder", self.encoder), ("actor", self.actor)):
            module_state = state.get(key)
            if not isinstance(module_state, dict):
                raise ValueError(f"actor snapshot is missing {key!r}")
            module.load_state_dict(
                {
                    name: torch.as_tensor(value, device=self.device)
                    for name, value in module_state.items()
                },
                strict=True,
            )
        self.encoder.eval()
        self.actor.eval()
