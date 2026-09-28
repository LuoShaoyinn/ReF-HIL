"""Shared UR5e action, safety, and camera-observation behavior."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from actor.robot.action import send_continuous_gripper_action
from actor.robot.rotation import apply_local_rotation_delta
from actor.robot.ur5e import RotationRecoveryConfig, UR5eBase, UR5eBaseConfig
from .manipulation_config import TaskConfig


@dataclass(kw_only=True)
class RobotConfig(UR5eBaseConfig):
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        resolution = self.task.global_camera_resolution
        if resolution is not None:
            width, height = resolution
            self.realsense = {
                **self.realsense,
                "global": replace(
                    self.realsense["global"],
                    width=int(width),
                    height=int(height),
                ),
            }
        self.gripper = replace(
            self.gripper,
            runtime_speed_limit=int(self.task.gripper_speed),
            runtime_force_limit=int(self.task.gripper_force),
        )
        self.impendence_control.configure_movable_axes(self.task.movable_axes)
        if not self.task.allow_rotation:
            self.impendence_control.rotation_recovery = None
            return
        self.impendence_control.rotation_recovery = RotationRecoveryConfig(
            nominal_rotvec=self.task.zero_point_rot,
            half_range_rotvec=self.task.safety_rot_half_range,
            inward_margin_rotvec=self.task.rotation_recovery_inward_margin,
            activation_delta_rotvec=self.task.rotation_recovery_activation_delta,
            max_wrench=self.task.rotation_recovery_max_wrench,
            wrench_gain_multiplier=self.task.rotation_recovery_wrench_gain_multiplier,
        )


def _crop(array: np.ndarray, clip: tuple[int, int, int, int], *, name: str) -> np.ndarray:
    x, y, width, height = clip
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid {name} crop {clip}")
    if y + height > array.shape[0] or x + width > array.shape[1]:
        raise ValueError(f"{name} crop {clip} is outside image shape {array.shape[:2]}")
    return array[y : y + height, x : x + width]


class Robot(UR5eBase):
    config: RobotConfig

    def gripper_action_range(self) -> tuple[int, int]:
        """Raw positions for action -1/+1, after startup hardware calibration."""
        return self.gripper.get_open_position(), self.gripper.get_closed_position()

    def connect(self) -> None:
        super().connect()
        if not self.config.task.allow_rotation:
            target_pose = self.impedance_control.get_actual_tcp_pose()
            target_pose[1] = np.asarray(
                self.config.task.zero_point_rot, dtype=np.float32
            )
            self.impedance_control.set_target_pose(target_pose)

    def apply_action(self, raw_action: dict) -> None:
        task = self.config.task
        tcp_pose = self.impedance_control.get_actual_tcp_pose()
        if task.allow_rotation and self.impedance_control.update_rotation_recovery(
            tcp_pose[1]
        ):
            # Strict recovery-only mode: ignore translation, gripper, and all
            # new rotation commands until the measured orientation is inside.
            return
        normalized_position = np.clip(
            np.asarray(raw_action["delta_pos"], dtype=np.float32).reshape(3), -1.0, 1.0
        )
        delta_pos = normalized_position * task.linear_speed
        delta_rot = np.asarray(raw_action["delta_rot"], dtype=np.float32).reshape(3)
        # Project the requested Cartesian target into the task workspace before
        # it reaches the impedance controller.
        tcp_pose[0] = np.clip(
            np.add(tcp_pose[0], delta_pos),
            np.asarray(task.safety_pos_min, dtype=np.float32),
            np.asarray(task.safety_pos_max, dtype=np.float32),
        )
        if task.allow_rotation:
            candidate = apply_local_rotation_delta(tcp_pose[1], delta_rot)
            tcp_pose[1] = self.impedance_control.constrain_rotation_target(
                candidate
            )
        else:
            # A translation-only policy cannot alter orientation. Keep all six
            # force-mode axes compliant and continuously track the task's
            # canonical upright pose through bounded rotational wrench.
            tcp_pose[1] = np.asarray(task.zero_point_rot, dtype=np.float32)
        self.impedance_control.set_target_pose(tcp_pose)
        if "gripper" in raw_action:
            send_continuous_gripper_action(
                self.gripper,
                float(raw_action["gripper"]),
                speed=task.gripper_speed,
                force=task.gripper_force,
            )

    def read_obs(self) -> dict:
        task = self.config.task
        global_image = self.cameras["global"].read_image()
        global_depth = self.cameras["global"].read_depth()
        wrist0_image = self.cameras["wrist_0"].read_image()
        wrist1_image = self.cameras["wrist_1"].read_image()
        if global_depth.shape[:2] != global_image.shape[:2]:
            raise ValueError(
                f"global RGB/depth size mismatch: {global_image.shape} vs {global_depth.shape}"
            )
        size = (task.policy_image_size, task.policy_image_size)
        images = {
            "global_0": cv2.resize(
                _crop(global_image, task.global_policy_clip, name="global policy"), size
            ),
            "global_0_depth": cv2.resize(
                _crop(global_depth, task.global_policy_clip, name="global policy depth"),
                size,
                interpolation=cv2.INTER_NEAREST,
            ),
            "wrist_0": cv2.resize(
                _crop(wrist0_image, task.wrist_0_policy_clip, name="wrist 0 policy"), size
            ),
            "wrist_1": cv2.resize(
                _crop(wrist1_image, task.wrist_1_policy_clip, name="wrist 1 policy"), size
            ),
        }
        for condition in task.success_conditions:
            images[condition.image_key] = _crop(
                global_image,
                condition.clip,
                name=f"success condition {condition.name!r}",
            ).copy()
            images[condition.depth_key] = _crop(
                global_depth,
                condition.clip,
                name=f"success condition {condition.name!r} depth",
            ).copy()
        if self.config.include_raw_images:
            images["global_raw"] = global_image.copy()
            images["global_depth_raw"] = global_depth.copy()
        return {
            "images": images,
            "tcp_pose": self.impedance_control.get_actual_tcp_pose().astype(np.float32),
            "tcp_speed": self.impedance_control.get_actual_tcp_speed().astype(np.float32),
            "tcp_force": self.impedance_control.get_actual_tcp_force().astype(np.float32),
            "gripper": np.float32(self.gripper.get_current_position()),
            "gripper_action_range": self.gripper_action_range(),
        }
