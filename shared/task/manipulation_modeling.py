"""Shared observation, Cartesian action, and reset implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
from torchvision import models as tv_models
from torchvision.transforms import functional as tvf

from shared.task.base import BaseModeling, ModelingConfig as BaseModelingConfig
from .manipulation_config import TaskConfig
from .gripper_calibration import gripper_control_position


def sample_reset_rotation_target(task: TaskConfig) -> np.ndarray:
    """Sample a gripper-local reset orientation around the nominal pose."""

    local_half_range = np.asarray(task.reset_rot_rnd_abs, dtype=np.float64)
    local_offset = np.random.uniform(
        -local_half_range, local_half_range
    )
    nominal = Rotation.from_rotvec(
        np.asarray(task.zero_point_rot, dtype=np.float64)
    )
    return (nominal * Rotation.from_rotvec(local_offset)).as_rotvec().astype(
        np.float32
    )


def local_rotation_error(
    current_rotvec: np.ndarray, target_rotvec: np.ndarray
) -> np.ndarray:
    """Return the local rotation that post-multiplies current into target."""

    current = Rotation.from_rotvec(
        np.asarray(current_rotvec, dtype=np.float64).reshape(3)
    )
    target = Rotation.from_rotvec(
        np.asarray(target_rotvec, dtype=np.float64).reshape(3)
    )
    return (current.inv() * target).as_rotvec().astype(np.float32)


class Resnet18Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


@dataclass(kw_only=True)
class ModelingConfig(BaseModelingConfig):
    obs_dims: int = field(init=False, default=1552)
    action_dims: int = field(init=False, default=4)
    mechanism_obs_dims: int = field(init=False, default=16)
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        self.obs_dims = self.task.obs_dims
        self.action_dims = self.task.action_dims
        self.mechanism_obs_dims = self.task.mechanism_obs_dims


class Modeling(BaseModeling):
    def __init__(self, config: ModelingConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.device = torch.device(self.config.device)
        self.image_encoder = Resnet18Encoder().to(self.device)
        self.image_encoder.eval()

    def build_obs(self, state: dict, info: dict, augment: bool = False) -> torch.Tensor:
        return self.build_observations([state], [info], augment=augment).squeeze(0)

    @staticmethod
    def _normalize(value: np.ndarray, value_range: np.ndarray) -> np.ndarray:
        normalized = 2.0 * (value - value_range[0]) / (value_range[1] - value_range[0]) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    def prepare_observation(self, raw_observation: dict) -> dict:
        task = self.config.task
        tcp_speed = np.asarray(raw_observation["tcp_speed"], dtype=np.float32).reshape(6)
        tcp_force = np.asarray(raw_observation["tcp_force"], dtype=np.float32).reshape(6)
        gripper_raw = float(np.asarray(raw_observation["gripper"]).reshape(()))
        tcp_rotvec = np.asarray(
            raw_observation["tcp_pose"], dtype=np.float32
        ).reshape(2, 3)[1]
        # Express world gravity in the gripper/TCP frame. This exposes tilt
        # without leaking absolute Cartesian position or yaw around gravity.
        projected_gravity = Rotation.from_rotvec(tcp_rotvec).inv().apply(
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        ).astype(np.float32)
        return {
            "tcp_speed": self._normalize(tcp_speed, np.asarray(task.tcp_speed_range, dtype=np.float32)),
            "tcp_force": self._normalize(tcp_force, np.asarray(task.tcp_force_range, dtype=np.float32)),
            "gripper": np.float32(
                self._normalize(
                    np.asarray([gripper_raw], dtype=np.float32),
                    np.asarray(task.gripper_position_range, dtype=np.float32)[:, None],
                )[0]
            ),
            "projected_gravity": projected_gravity,
            "images": raw_observation["images"],
        }

    def build_observations(
        self, states: list[dict], infos: list[dict], *, augment: bool = False
    ) -> torch.Tensor:
        assert len(states) == len(infos)
        if not states:
            return torch.empty((0, self.config.obs_dims), dtype=torch.float32, device=self.device)
        task = self.config.task
        images = np.stack(
            [state["images"][key] for state in states for key in task.image_keys], axis=0
        )
        image_batch = torch.from_numpy(images).to(self.device, non_blocking=True)
        image_batch = image_batch.permute(0, 3, 1, 2).float().div_(255.0)
        del augment
        image_batch = tvf.normalize(
            image_batch, mean=list(task.imagenet_mean), std=list(task.imagenet_std)
        )
        with torch.no_grad():
            image_latent = self.image_encoder(image_batch).reshape(len(states), -1)
        proprio = torch.stack(
            [self.build_proprio(state, info) for state, info in zip(states, infos, strict=True)], dim=0
        )
        return torch.cat((proprio, image_latent.detach()), dim=1)

    def build_proprio(self, raw_observation: dict, info: dict) -> torch.Tensor:
        del info
        tcp_speed = np.asarray(raw_observation["tcp_speed"], dtype=np.float32).reshape(6)
        tcp_force = np.asarray(raw_observation["tcp_force"], dtype=np.float32).reshape(6)
        gripper = np.asarray([raw_observation["gripper"]], dtype=np.float32)
        projected_gravity = np.asarray(
            raw_observation["projected_gravity"], dtype=np.float32
        ).reshape(3)
        proprio = np.concatenate(
            (tcp_speed, tcp_force, gripper, projected_gravity),
            axis=0,
        ).astype(np.float32)
        expected = self.config.mechanism_obs_dims
        if proprio.shape != (expected,) or not np.isfinite(proprio).all():
            raise ValueError(
                f"stored proprio must be a finite normalized {expected}D vector"
            )
        return torch.from_numpy(proprio).to(self.device)

    def build_action(self, raw_action: dict, info: dict, augment: bool = False) -> torch.Tensor:
        del info, augment
        action = np.concatenate(
            (np.asarray(raw_action["delta_pos"], dtype=np.float32),
             np.asarray([raw_action["gripper"]], dtype=np.float32)), axis=0
        )
        return torch.from_numpy(action).to(self.device).clamp_(-1.0, 1.0)

    def build_actions(self, raw_actions: list[dict], infos: list[dict], *, augment: bool = False) -> torch.Tensor:
        del infos, augment
        delta_pos = np.stack([action["delta_pos"] for action in raw_actions], axis=0)
        gripper = np.asarray([action["gripper"] for action in raw_actions], dtype=np.float32).reshape(-1, 1)
        action = np.concatenate((delta_pos, gripper), axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(action).to(self.device, non_blocking=True).clamp_(-1.0, 1.0)

    def build_action_from_spacemouse(self, raw_action: dict, info: dict) -> torch.Tensor:
        del info
        action = np.concatenate(
            (np.asarray(raw_action["delta_pos"], dtype=np.float32),
             np.asarray([raw_action["gripper"]], dtype=np.float32)), axis=0
        )
        return torch.from_numpy(action).to(self.device).clamp_(-1.0, 1.0)

    def build_gripper_operator_request(self, raw_observation: dict) -> dict:
        task = self.config.task
        return {
            "gripper_position": gripper_control_position(
                raw_observation, task.gripper_position_range
            ),
        }

    def build_operator_request(self, raw_observation: dict) -> dict:
        task = self.config.task
        target_rot = (
            task.zero_point_rot
            if task.spacemouse_copilot_target_rot is None
            else task.spacemouse_copilot_target_rot
        )
        return {
            **self.build_gripper_operator_request(raw_observation),
            "tcp_pose": np.asarray(raw_observation["tcp_pose"], dtype=np.float32),
            "rotation_copilot": {
                "target_rotvec": np.asarray(
                    target_rot, dtype=np.float32
                ),
                "angular_speed": task.angular_speed,
                "gain": task.spacemouse_copilot_gain,
                "deadband_rad": task.spacemouse_copilot_deadband_rad,
            },
        }

    def parse_action(self, action: torch.Tensor) -> dict:
        return {
            "delta_pos": action[:3].detach().cpu().numpy(),
            "delta_rot": np.zeros(3, dtype=np.float32),
            "gripper": float(action[3].detach().cpu().item()),
        }

    def select_action(self, policy_action: torch.Tensor, human_action: dict, last_gripper_command: float) -> tuple[torch.Tensor, bool]:
        if not self.operator_has_intent(human_action):
            return policy_action, False
        resolved = self.prepare_spacemouse_action(
            human_action, last_gripper_command
        )
        return self.build_action_from_spacemouse(resolved, {}), True

    def reset(self, send_action: object, read_observation: object) -> None:
        task = self.config.task

        def move(
            delta_pos: np.ndarray,
            delta_rot: np.ndarray,
            duration: float,
        ) -> None:
            send_action({
                "delta_pos": np.clip(delta_pos / task.linear_speed, -1.0, 1.0).astype(np.float32),
                "delta_rot": np.clip(
                    delta_rot / task.angular_speed, -1.0, 1.0
                ).astype(np.float32),
                "gripper": task.reset_gripper,
            })
            time.sleep(duration)

        def move_to(
            target_pos: np.ndarray,
            target_rot: np.ndarray | None,
        ) -> None:
            start = time.time()
            while True:
                current_pose = np.asarray(
                    read_observation()["tcp_pose"], dtype=np.float32
                ).reshape(2, 3)
                position_error = target_pos - current_pose[0]
                rotation_error = (
                    np.zeros(3, dtype=np.float32)
                    if target_rot is None
                    else local_rotation_error(current_pose[1], target_rot)
                )
                position_done = (
                    np.linalg.norm(position_error)
                    <= task.reset_position_tolerance
                )
                rotation_done = target_rot is None or (
                    np.linalg.norm(rotation_error)
                    <= task.reset_rotation_tolerance
                )
                if position_done and rotation_done:
                    break
                move(
                    position_error,
                    rotation_error,
                    task.reset_step_duration_s,
                )
                if time.time() - start > task.reset_move_timeout_s:
                    break

        reset_pos = np.asarray(task.reset_pos, dtype=np.float32).copy()
        reset_pos += np.random.uniform(
            -np.asarray(task.reset_rnd_abs, dtype=np.float32),
            np.asarray(task.reset_rnd_abs, dtype=np.float32),
        ).astype(np.float32)
        reset_rot = (
            sample_reset_rotation_target(task)
            if task.allow_rotation
            else None
        )
        move(
            np.asarray(task.reset_lift_delta, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            task.reset_lift_duration_s,
        )
        move_to(reset_pos, reset_rot)
        if task.reset_settle_time_s > 0.0:
            time.sleep(task.reset_settle_time_s)
