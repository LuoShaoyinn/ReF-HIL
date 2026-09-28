from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from shared.zmq import DictMessage


@dataclass(kw_only=True)
class PolicyConfig:
    device: str = "cuda"


class BasePolicy(ABC):
    """Algorithm policy interface shared by actor and learner."""

    def __init__(self, *, config: PolicyConfig) -> None:
        self.config = config

    @abstractmethod
    def sample_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def update(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict:
        '''output dict to be logged'''
        pass

    @abstractmethod
    def export(self) -> DictMessage:
        pass

    @abstractmethod
    def load(self, state: dict) -> None:
        pass
