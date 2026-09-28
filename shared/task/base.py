from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from shared.zmq import DictMessage


@dataclass(kw_only=True)
class ModelingConfig:
    obs_dims: int
    action_dims: int
    device: str = "cuda"


class BaseModeling(ABC):
    def __init__(self, *, config: ModelingConfig) -> None:
        self.config = config

    def prepare_observation(self, raw_observation: DictMessage) -> DictMessage:
        """Convert a hardware observation into the actor/replay wire format."""
        return raw_observation

    @abstractmethod
    def build_obs(self, state: DictMessage, info: DictMessage, augment: bool = False) -> torch.Tensor:
        '''Build observation for policy from a raw state dict'''

    def build_observations(
        self,
        states: list[DictMessage],
        infos: list[DictMessage],
        *,
        augment: bool = False,
    ) -> torch.Tensor:
        """Build a batch of observations.

        Tasks with an image encoder should override this method.  The fallback
        deliberately preserves the scalar ``build_obs`` contract for small
        task implementations and tests.
        """
        assert len(states) == len(infos)
        return torch.stack(
            [self.build_obs(state, info, augment=augment) for state, info in zip(states, infos, strict=True)],
            dim=0,
        )

    @abstractmethod
    def build_action(self, raw_action: DictMessage, info: DictMessage, augment: bool = False) -> torch.Tensor:
        '''where the np array is build into action in dict'''

    def build_actions(
        self,
        raw_actions: list[DictMessage],
        infos: list[DictMessage],
        *,
        augment: bool = False,
    ) -> torch.Tensor:
        """Batch counterpart to ``build_action`` with a safe scalar fallback."""
        assert len(raw_actions) == len(infos)
        return torch.stack(
            [self.build_action(action, info, augment=augment) for action, info in zip(raw_actions, infos, strict=True)],
            dim=0,
        )

    def build_action_from_spacemouse(self, raw_action: dict, info: dict) -> torch.Tensor:
        '''where the np array is build into action in dict'''

    def build_operator_request(self, raw_observation: DictMessage) -> DictMessage:
        """Build task context sent to the independent human-input process."""
        del raw_observation
        return {}

    @staticmethod
    def prepare_spacemouse_action(
        raw_action: DictMessage,
        last_gripper_command: float,
    ) -> DictMessage:
        """Preserve the absolute gripper command resolved by the operator."""

        action = dict(raw_action)
        if "gripper" in action:
            command = action["gripper"]
        elif action.get("gripper_open_pressed", False):
            command = -1.0
        elif action.get("gripper_close_pressed", False):
            command = 1.0
        elif action.get("gripper_position") is not None:
            position = float(
                np.clip(action["gripper_position"], 0.0, 1.0)
            )
            command = 2.0 * position - 1.0
        else:
            command = last_gripper_command
        action["gripper"] = float(np.clip(command, -1.0, 1.0))
        return action

    def operator_action_has_intent(self, raw_action: DictMessage) -> bool:
        """Return whether the scaled operator action should seize control."""
        translation = np.asarray(
            raw_action.get("delta_pos", np.zeros(3)), dtype=np.float32
        ).reshape(3)
        rotation = np.asarray(
            raw_action.get("delta_rot", np.zeros(3)), dtype=np.float32
        ).reshape(3)
        return bool(
            np.linalg.norm(translation) > 1e-3
            or np.linalg.norm(rotation) > 1e-3
            or raw_action.get("gripper_pressed", False)
            or raw_action.get("gripper_close_pressed", False)
            or raw_action.get("gripper_open_pressed", False)
        )

    def operator_has_intent(self, raw_action: DictMessage) -> bool:
        """Honor the operator's raw six-axis/button intervention decision."""

        return bool(
            raw_action.get("is_intervene", False)
            or self.operator_action_has_intent(raw_action)
        )

    @abstractmethod
    def select_action(
        self,
        policy_action: torch.Tensor,
        human_action: dict,
        last_gripper_command: float,
    ) -> tuple[torch.Tensor, bool]:
        pass

    @abstractmethod
    def parse_action(self, action: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def reset(self, send_action: object | None = None, read_observation: object | None = None) -> None:
        pass

    def build_info(self,
                   is_intervene: bool,
                   steps_in_episode: int,
                   **kwargs: object) -> DictMessage:
        info: DictMessage = {
            "is_intervene": bool(is_intervene),
            "steps_in_episode": int(steps_in_episode),
        }
        info.update(kwargs)
        return info
