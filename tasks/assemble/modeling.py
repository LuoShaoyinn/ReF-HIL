from __future__ import annotations

from dataclasses import dataclass, field

from shared.task.manipulation_modeling import Modeling as BaseModeling
from shared.task.manipulation_modeling import ModelingConfig as BaseModelingConfig
from .config import TaskConfig
from .reset import ResetConfig, run_reset


@dataclass(kw_only=True)
class ModelingConfig(BaseModelingConfig):
    task: TaskConfig = field(default_factory=TaskConfig)


class Modeling(BaseModeling):
    config: ModelingConfig

    def reset(self, send_action: object, read_observation: object) -> None:
        run_reset(
            send_action,
            read_observation,
            config=ResetConfig(
                safety_pos_min=self.config.task.safety_pos_min,
                safety_pos_max=self.config.task.safety_pos_max,
                linear_speed=self.config.task.linear_speed,
            ),
        )
