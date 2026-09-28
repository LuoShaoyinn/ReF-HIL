from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.zmq import DictMessage
from .base import BaseClassifier, BaseClassifierConfig


@dataclass(kw_only=True)
class ResNetClassifierConfig(BaseClassifierConfig):
    device: str = "cuda"
    success_threshold: float = 0.5
    resnet_name: str = "resnet18"
    pretrained: bool = False
    in_channels: int = 3


class ResNetClassifier(BaseClassifier):
    def __init__(self, *, config: ResNetClassifierConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.device = torch.device(self.config.device)
        self.model = self.build_model().to(self.device)

    @abstractmethod
    def build_model(self) -> nn.Module:
        pass

    @property
    def checkpoint_path(self) -> Path:
        return Path("outputs") / str(self.config.experiment_name) / "classifier.pt"

    def preprocessing_contract(self) -> dict[str, object]:
        return {
            "resnet_name": str(self.config.resnet_name),
            "in_channels": int(self.config.in_channels),
        }

    def save(self, checkpoint_path: Path | str | None = None) -> Path:
        target = self.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "preprocessing": self.preprocessing_contract(),
            },
            target,
        )
        return target

    def load(self, checkpoint_path: Path | str | None = None, *, strict: bool = False) -> None:
        source = self.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
        payload = torch.load(source, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            checkpoint_contract = payload.get("preprocessing")
            runtime_contract = self.preprocessing_contract()
            if checkpoint_contract != runtime_contract:
                raise ValueError(
                    "classifier preprocessing mismatch: "
                    f"checkpoint={checkpoint_contract}, runtime={runtime_contract}"
                )
            state_dict = payload["state_dict"]
        else:
            # Legacy checkpoints contain only the state dict. Their input
            # geometry cannot be inferred, so task config remains authoritative.
            state_dict = payload
        self.model.load_state_dict(state_dict, strict=strict)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device).float())

    def inference(self, *, raw_robot_observation: DictMessage) -> bool:
        score = self.score(raw_robot_observation=raw_robot_observation)
        return bool(score >= float(self.config.success_threshold))

    def score(self, *, raw_robot_observation: DictMessage) -> float:
        """Return the calibrated sigmoid output used by the success threshold."""

        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(self.modeling(raw_robot_observation), dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.forward_logits(x).reshape(-1)
            score = float(torch.sigmoid(logits[0]).item())
        return score

    @abstractmethod
    def modeling(self, raw_robot_observation: DictMessage) -> torch.Tensor | np.ndarray:
        """Return one sample image in CHW or BCHW."""
        pass
