from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from shared.zmq import DictMessage


@dataclass(kw_only=True)
class BaseClassifierConfig:
    experiment_name: str = "default_experiment"


class BaseClassifier(ABC):
    def __init__(self, *, config: BaseClassifierConfig) -> None:
        self.config = config

    @abstractmethod
    def inference(self, *, raw_robot_observation: DictMessage) -> bool:
        pass
