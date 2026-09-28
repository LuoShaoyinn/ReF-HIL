"""Knob observation, five-dimensional action, and reset contract."""

from __future__ import annotations

import math
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
    action_dims: int = field(init=False, default=5)
    task: TaskConfig = field(default_factory=TaskConfig)


def rotation_at_yaw(task: TaskConfig, yaw: float) -> np.ndarray:
    nominal = Rotation.from_rotvec(
        np.asarray(task.zero_point_rot, dtype=np.float64)
    )
    local_yaw = Rotation.from_rotvec(
        np.asarray([0.0, 0.0, float(yaw)], dtype=np.float64)
    )
    return (nominal * local_yaw).as_rotvec().astype(np.float32)


def yaw_relative_to_nominal(task: TaskConfig, rotvec: np.ndarray) -> float:
    nominal = Rotation.from_rotvec(
        np.asarray(task.zero_point_rot, dtype=np.float64)
    )
    actual = Rotation.from_rotvec(
        np.asarray(rotvec, dtype=np.float64).reshape(3)
    )
    relative = nominal.inv() * actual
    matrix = relative.as_matrix()
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


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
        return np.concatenate(
            (
                delta_pos,
                np.asarray(
                    [delta_rot[2], raw_action.get("gripper", -1.0)],
                    dtype=np.float32,
                ),
            )
        )

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
        actions = np.stack(
            [self._policy_action(raw_action) for raw_action in raw_actions]
        )
        return torch.from_numpy(actions).to(
            self.device, non_blocking=True
        ).clamp_(-1.0, 1.0)

    def build_action_from_spacemouse(
        self, raw_action: dict, info: dict
    ) -> torch.Tensor:
        return self.build_action(raw_action, info)

    def parse_action(self, action: torch.Tensor) -> dict:
        values = action.detach().cpu().numpy().reshape(5)
        return {
            "delta_pos": values[:3].astype(np.float32, copy=True),
            "delta_rot": np.asarray(
                [0.0, 0.0, values[3]], dtype=np.float32
            ),
            "gripper": float(values[4]),
        }

    def select_action(
        self,
        policy_action: torch.Tensor,
        human_action: dict,
        last_gripper_command: float,
    ) -> tuple[torch.Tensor, bool]:
        if not self.operator_has_intent(human_action):
            return policy_action, False
        resolved = self.prepare_spacemouse_action(
            human_action, last_gripper_command
        )
        return self.build_action_from_spacemouse(resolved, {}), True

    def operator_action_has_intent(self, raw_action: dict) -> bool:
        return super().operator_action_has_intent(raw_action)

    def build_operator_request(self, raw_observation: dict) -> dict:
        # Knob uses manually commanded local yaw while roll/pitch are held by
        # the task robot, so the generic all-axis rotation copilot is disabled.
        return self.build_gripper_operator_request(raw_observation)

    def reset(self, send_action: object, read_observation: object) -> None:
        reset_started = time.monotonic()
        task = self.config.task
        current_pose = np.asarray(
            read_observation()["tcp_pose"], dtype=np.float32
        ).reshape(2, 3)

        # Stage 1: open before any reset motion while preserving the measured
        # pose. The gripper SDK command is synchronous at the robot boundary.
        send_action(
            {
                "gripper_only": True,
                "gripper": float(task.reset_gripper),
            }
        )

        # Stage 2: move only Z while preserving the measured orientation.
        z_reset_pose = current_pose.copy()
        z_reset_pose[0, 2] = float(task.reset_clearance_z)
        if not self._move_reset_pose(
            send_action,
            read_observation,
            z_reset_pose,
            position_axes=(False, False, True),
            check_rotation=False,
        ):
            send_action({"hold_current_pose": True})
            raise RuntimeError("rotate_knob Z reset timed out; check the fixture")

        # Stage 3: after Z is safe, rotate in place through increasing yaw.
        current_pose = np.asarray(
            read_observation()["tcp_pose"], dtype=np.float32
        ).reshape(2, 3)
        try:
            rotation_complete = self._rotate_to_reset_yaw_increasing(
                send_action, read_observation, current_pose
            )
        except RuntimeError:
            send_action({"hold_current_pose": True})
            raise
        if not rotation_complete:
            send_action({"hold_current_pose": True})
            raise RuntimeError("rotate_knob rotation reset timed out; check the fixture")

        # Stage 4: move only XY to the randomized reset point. Its Z equals the
        # completed Stage-3 target, so Z and reset yaw remain fixed.
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
            [target_pos, rotation_at_yaw(task, task.reset_yaw_rad)],
            dtype=np.float32,
        )
        if not self._move_reset_pose(
            send_action,
            read_observation,
            target_pose,
            position_axes=(True, True, False),
        ):
            send_action({"hold_current_pose": True})
            raise RuntimeError("rotate_knob pose reset timed out; check the fixture")
        completed = int(getattr(self, "_completed_resets", 0))
        self._completed_resets = completed + 1
        if completed:
            robot_ready = time.monotonic()
            deadline = max(
                reset_started + task.reset_total_seconds,
                robot_ready + task.human_reset_seconds,
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                print(f"[rotate_knob] manually reset knob: {remaining:.1f}s remaining", flush=True)
                time.sleep(min(1.0, remaining))

    def _rotate_to_reset_yaw_increasing(
        self,
        send_action: object,
        read_observation: object,
        current_pose: np.ndarray,
    ) -> bool:
        """Reach reset yaw through numerically increasing local-yaw waypoints.

        A direct -100 to +100 degree rotation target is a 200-degree positive
        turn, so an SO(3) controller would otherwise choose the opposite
        160-degree shortest path. Waypoints keep every commanded increment at
        most ``reset_yaw_waypoint_step_rad`` and therefore preserve the
        intended direction without coupling reset behavior to online actions.
        """

        task = self.config.task
        current_pose = np.asarray(current_pose, dtype=np.float32).reshape(2, 3)
        current_yaw = float(np.clip(
            yaw_relative_to_nominal(task, current_pose[1]),
            *task.yaw_range_rad,
        ))
        target_yaw = float(task.reset_yaw_rad)
        tolerance = float(task.reset_rotation_tolerance)
        if current_yaw > target_yaw + tolerance:
            raise RuntimeError(
                "rotate_knob cannot reach reset yaw by increasing local yaw: "
                f"current={math.degrees(current_yaw):.1f} deg, "
                f"target={math.degrees(target_yaw):.1f} deg"
            )

        step = min(float(task.reset_yaw_waypoint_step_rad), math.pi / 2)
        waypoint_yaws: list[float] = []
        waypoint_yaw = current_yaw + step
        while waypoint_yaw < target_yaw - tolerance:
            waypoint_yaws.append(waypoint_yaw)
            waypoint_yaw += step
        waypoint_yaws.append(target_yaw)

        for yaw in waypoint_yaws:
            waypoint_pose = current_pose.copy()
            waypoint_pose[1] = rotation_at_yaw(task, yaw)
            if not self._move_reset_pose(
                send_action,
                read_observation,
                waypoint_pose,
                position_axes=(False, False, False),
            ):
                return False
        return True

    def _move_reset_pose(
        self,
        send_action: object,
        read_observation: object,
        target_pose: np.ndarray,
        *,
        position_axes: tuple[bool, bool, bool] = (True, True, True),
        check_rotation: bool = True,
    ) -> bool:
        task = self.config.task
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(2, 3)
        active_position_axes = np.asarray(position_axes, dtype=bool)
        if active_position_axes.shape != (3,):
            raise ValueError("position_axes must contain XYZ booleans")
        started = time.monotonic()
        while True:
            observation = read_observation()
            current_pose = np.asarray(
                observation["tcp_pose"], dtype=np.float32
            ).reshape(2, 3)
            position_error = target_pose[0] - current_pose[0]
            position_error[~active_position_axes] = 0.0
            rotation_error = (
                Rotation.from_rotvec(current_pose[1]).inv()
                * Rotation.from_rotvec(target_pose[1])
            ).as_rotvec()
            if not check_rotation:
                rotation_error.fill(0.0)
            position_error_norm = float(np.linalg.norm(position_error))
            rotation_error_norm = float(np.linalg.norm(rotation_error))
            position_done = position_error_norm <= task.reset_position_tolerance
            rotation_done = rotation_error_norm <= task.reset_rotation_tolerance
            if position_done and rotation_done:
                return True
            elapsed = time.monotonic() - started
            if elapsed >= task.reset_stage_timeout_s:
                return False
            # The gripper is opened once, synchronously, at the start of reset.
            # Re-sending the same target at the pose-control rate makes the
            # Robotiq repeatedly restart its motion and can cause chatter.
            send_action(
                {
                    "reset_target_pose": target_pose.copy(),
                    "reset_position_axes": tuple(
                        bool(value) for value in active_position_axes
                    ),
                    "reset_rotation": bool(check_rotation),
                }
            )
            time.sleep(
                min(task.reset_step_duration_s, task.reset_stage_timeout_s - elapsed)
            )
