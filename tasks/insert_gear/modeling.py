"""Gear-insertion observation, action, and reset contract."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from shared.task.manipulation_modeling import Modeling as BaseModeling
from shared.task.manipulation_modeling import ModelingConfig as BaseModelingConfig
from tasks.gear.modeling import rotation_at_yaw, yaw_relative_to_nominal

from .config import TaskConfig


@dataclass(kw_only=True)
class ModelingConfig(BaseModelingConfig):
    action_dims: int = field(init=False, default=4)
    task: TaskConfig = field(default_factory=TaskConfig)


class Modeling(BaseModeling):
    config: ModelingConfig

    @staticmethod
    def _policy_action(raw_action: dict) -> np.ndarray:
        delta_pos = np.asarray(
            raw_action.get("delta_pos", np.zeros(3)), dtype=np.float32
        ).reshape(3)
        delta_rot = np.asarray(
            raw_action.get("delta_rot", np.zeros(3)), dtype=np.float32
        ).reshape(3)
        return np.concatenate((delta_pos, np.asarray([delta_rot[2]], dtype=np.float32)))

    def build_action(self, raw_action: dict, info: dict, augment: bool = False) -> torch.Tensor:
        del info, augment
        return torch.from_numpy(self._policy_action(raw_action)).to(self.device).clamp_(-1.0, 1.0)

    def build_actions(
        self, raw_actions: list[dict], infos: list[dict], *, augment: bool = False
    ) -> torch.Tensor:
        del infos, augment
        actions = np.stack([self._policy_action(action) for action in raw_actions])
        return torch.from_numpy(actions).to(self.device, non_blocking=True).clamp_(-1.0, 1.0)

    def build_action_from_spacemouse(self, raw_action: dict, info: dict) -> torch.Tensor:
        return self.build_action(raw_action, info)

    def parse_action(self, action: torch.Tensor) -> dict:
        values = action.detach().cpu().numpy().reshape(4)
        return {
            "delta_pos": values[:3].astype(np.float32, copy=True),
            "delta_rot": np.asarray([0.0, 0.0, values[3]], dtype=np.float32),
            "gripper": float(self.config.task.fixed_gripper_action),
        }

    def select_action(
        self, policy_action: torch.Tensor, human_action: dict, last_gripper_command: float
    ) -> tuple[torch.Tensor, bool]:
        del last_gripper_command
        if not self.operator_has_intent(human_action):
            return policy_action, False
        return self.build_action_from_spacemouse(human_action, {}), True

    def build_operator_request(self, raw_observation: dict) -> dict:
        del raw_observation
        return {}

    def reset(self, send_action: object, read_observation: object) -> None:
        task = self.config.task
        current_pose = np.asarray(
            read_observation()["tcp_pose"], dtype=np.float32
        ).reshape(2, 3)
        def move(pose, stage, gripper):
            if not self._move_reset_pose(send_action, read_observation, pose, gripper=gripper):
                raise RuntimeError(f"insert_gear reset failed at {stage}")

        def grip(value, *, wait_s):
            send_action({"insert_gear_reset": True, "gripper_only": True, "gripper": value})
            time.sleep(wait_s)

        closed = float(task.fixed_gripper_action)
        if current_pose[0, 2] > np.float32(task.reset_release_z):
            release_pose = current_pose.copy()
            release_pose[0, 2] = task.reset_release_z
            move(release_pose, "release descent", closed)
        grip(-1.0, wait_s=task.reset_gripper_wait_s)
        # Use the actual release XY/orientation; align rotation only after lifting.
        lift_pose = np.asarray(read_observation()["tcp_pose"], dtype=np.float32).reshape(2, 3).copy()
        lift_pose[0, 2] = task.reset_pos[2]
        move(lift_pose, "open-gripper lift", -1.0)
        lift_pose[1] = rotation_at_yaw(task, task.startup_pickup_yaw_rad)
        move(lift_pose, "rotation alignment", -1.0)
        print(f"[insert_gear] Reposition the gear at the pickup point; waiting {task.reset_manual_wait_s:g}s.", flush=True)
        time.sleep(task.reset_manual_wait_s)

        pickup = np.asarray([task.startup_pickup_pos, rotation_at_yaw(task, task.startup_pickup_yaw_rad)], dtype=np.float32)
        approach = pickup.copy()
        approach[0, 2] = task.reset_pos[2]
        move(approach, "pickup approach", -1.0)
        move(pickup, "pickup descent", -1.0)
        grip(closed, wait_s=task.reset_gripper_close_wait_s)
        move(approach, "closed-gripper lift", closed)

        target_pos = np.asarray(task.reset_pos, dtype=np.float32)
        target_pos += np.random.uniform(
            -np.asarray(task.reset_rnd_abs, dtype=np.float32),
            np.asarray(task.reset_rnd_abs, dtype=np.float32),
        ).astype(np.float32)
        target_pos = np.clip(
            target_pos,
            np.asarray(task.safety_pos_min, dtype=np.float32),
            np.asarray(task.safety_pos_max, dtype=np.float32),
        )
        target_pose = np.asarray(
            [target_pos, rotation_at_yaw(task, task.reset_yaw_rad)], dtype=np.float32
        )
        move(target_pose, "randomized start", closed)
        time.sleep(task.reset_settle_time_s)

    def _move_reset_pose(
        self,
        send_action: object,
        read_observation: object,
        target_pose: np.ndarray,
        *,
        gripper: float = 1.0,
    ) -> bool:
        task = self.config.task
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(2, 3)
        started = time.monotonic()
        while True:
            current_pose = np.asarray(read_observation()["tcp_pose"], dtype=np.float32).reshape(2, 3)
            position_error = target_pose[0] - current_pose[0]
            rotation_error = (
                Rotation.from_rotvec(current_pose[1]).inv()
                * Rotation.from_rotvec(target_pose[1])
            ).as_rotvec()
            if (
                np.linalg.norm(position_error) <= task.reset_position_tolerance
                and np.linalg.norm(rotation_error) <= task.reset_rotation_tolerance
            ):
                return True
            elapsed = time.monotonic() - started
            if elapsed >= task.reset_stage_timeout_s:
                return False
            action = {"reset_target_pose": target_pose.copy(),
                      "insert_gear_reset": True, "gripper": float(gripper)}
            send_action(action)
            time.sleep(min(task.reset_step_duration_s, task.reset_stage_timeout_s - elapsed))
