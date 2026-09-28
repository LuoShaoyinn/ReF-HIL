"""Push-T observation, XYZ action, teleoperation, and reset contract."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from shared.task.manipulation_modeling import Modeling as BaseModeling
from shared.task.manipulation_modeling import ModelingConfig as BaseModelingConfig

from .config import TaskConfig


@dataclass(kw_only=True)
class ModelingConfig(BaseModelingConfig):
    action_dims: int = field(init=False, default=3)
    task: TaskConfig = field(default_factory=TaskConfig)


class Modeling(BaseModeling):
    config: ModelingConfig

    @staticmethod
    def _policy_action(raw_action: dict) -> np.ndarray:
        delta_pos = np.asarray(
            raw_action.get("delta_pos", np.zeros(3)), dtype=np.float32
        ).reshape(3)
        return delta_pos

    def build_action(
        self, raw_action: dict, info: dict, augment: bool = False
    ) -> torch.Tensor:
        del info, augment
        return torch.from_numpy(self._policy_action(raw_action)).to(
            self.device
        ).clamp_(-1.0, 1.0)

    def build_actions(
        self,
        raw_actions: list[dict],
        infos: list[dict],
        *,
        augment: bool = False,
    ) -> torch.Tensor:
        del infos, augment
        action = np.stack(
            [self._policy_action(raw_action) for raw_action in raw_actions]
        )
        return torch.from_numpy(action).to(
            self.device, non_blocking=True
        ).clamp_(-1.0, 1.0)

    def build_action_from_spacemouse(
        self, raw_action: dict, info: dict
    ) -> torch.Tensor:
        return self.build_action(raw_action, info)

    def parse_action(self, action: torch.Tensor) -> dict:
        values = action.detach().cpu().numpy().reshape(3)
        return {
            "delta_pos": values.astype(np.float32, copy=False),
            "delta_rot": np.zeros(3, dtype=np.float32),
            # Fixed hardware command only; this is not represented in the
            # three-dimensional policy action.
            "gripper": float(self.config.task.fixed_gripper_action),
        }

    def select_action(
        self,
        policy_action: torch.Tensor,
        human_action: dict,
        last_gripper_command: float,
    ) -> tuple[torch.Tensor, bool]:
        del last_gripper_command
        if not self.operator_has_intent(human_action):
            return policy_action, False
        return self.build_action_from_spacemouse(human_action, {}), True

    def build_operator_request(self, raw_observation: dict) -> dict:
        # Push-T uses manual local-z rotation. Do not expose the generic
        # all-axis upright copilot for this task.
        del raw_observation
        return {}

    def operator_action_has_intent(self, raw_action: dict) -> bool:
        return super().operator_action_has_intent(raw_action)

    def reset(self, send_action: object, read_observation: object) -> None:
        task = self.config.task
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
        target_rot = np.asarray(task.zero_point_rot, dtype=np.float32)

        current_pose = np.asarray(
            read_observation()["tcp_pose"], dtype=np.float32
        ).reshape(2, 3)
        reset_start_pos = np.clip(
            current_pose[0],
            np.asarray(task.safety_pos_min, dtype=np.float32),
            np.asarray(task.safety_pos_max, dtype=np.float32),
        )
        # First retreat along Y at the current height. This order is specific
        # to the physical Push-T fixture and intentionally precedes lifting.
        y_target_pose = np.asarray(
            [
                reset_start_pos,
                target_rot,
            ],
            dtype=np.float32,
        )
        y_target_pose[0, 1] = target_pos[1]
        if not self._move_reset_pose(
            send_action,
            read_observation,
            y_target_pose,
            timeout_s=task.reset_move_timeout_s,
        ):
            raise TimeoutError("push_t reset timed out while moving along Y")

        lift_pose = y_target_pose.copy()
        lift_pose[0, 2] = task.reset_lift_z
        if not self._move_reset_pose(
            send_action,
            read_observation,
            lift_pose,
            timeout_s=task.reset_move_timeout_s,
        ):
            raise TimeoutError("push_t reset timed out while lifting")

        target_pose = np.asarray([target_pos, target_rot], dtype=np.float32)
        # The final approach is deliberately bounded in wall-clock time. If
        # contact prevents exact convergence, the impedance target remains at
        # the sampled pose when the episode begins.
        self._move_reset_pose(
            send_action,
            read_observation,
            target_pose,
            timeout_s=task.reset_target_move_timeout_s,
        )
        if task.reset_settle_time_s > 0.0:
            print(
                f"[push_t] place the T object: "
                f"waiting {task.reset_settle_time_s:g}s",
                flush=True,
            )
            time.sleep(task.reset_settle_time_s)
            print("[push_t] placement time complete; starting episode", flush=True)

    def _move_reset_pose(
        self,
        send_action: object,
        read_observation: object,
        target_pose: np.ndarray,
        *,
        timeout_s: float,
    ) -> bool:
        task = self.config.task
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(2, 3)

        started = time.monotonic()
        while True:
            current_pose = np.asarray(
                read_observation()["tcp_pose"], dtype=np.float32
            ).reshape(2, 3)
            pos_error = target_pose[0] - current_pose[0]
            current_rotation = Rotation.from_rotvec(current_pose[1])
            desired_rotation = Rotation.from_rotvec(target_pose[1])
            rotation_error = (
                current_rotation.inv() * desired_rotation
            ).as_rotvec()
            if (
                np.linalg.norm(pos_error) <= task.reset_position_tolerance
                and np.linalg.norm(rotation_error)
                <= task.reset_rotation_tolerance
            ):
                return True
            elapsed = time.monotonic() - started
            if elapsed >= timeout_s:
                return False
            send_action({
                "reset_target_pose": target_pose.copy(),
                "gripper": float(task.fixed_gripper_action),
            })
            time.sleep(min(task.reset_step_duration_s, timeout_s - elapsed))
