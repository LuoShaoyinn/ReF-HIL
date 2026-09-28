"""Knob robot control: XYZ translation, local yaw, and continuous gripper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from actor.robot.action import send_continuous_gripper_action
from actor.robot.ur5e import RotationRecoveryConfig
from shared.task.manipulation_robot import Robot as BaseRobot
from shared.task.manipulation_robot import RobotConfig as BaseRobotConfig
from .config import TaskConfig
from .modeling import rotation_at_yaw, yaw_relative_to_nominal


@dataclass(kw_only=True)
class RobotConfig(BaseRobotConfig):
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        task = self.task
        # Nominal tool Z is aligned with base Z (opposite sign) in this task.
        # Raise only the yaw speed cap, not translational limits or torques.
        self.impendence_control.limit["max_speed"][1, 2] = task.yaw_speed_limit_rad_s
        yaw_min, yaw_max = task.yaw_range_rad
        yaw_center = 0.5 * (yaw_min + yaw_max)
        yaw_half_range = 0.5 * (yaw_max - yaw_min)
        self.impendence_control.rotation_recovery = RotationRecoveryConfig(
            nominal_rotvec=rotation_at_yaw(task, yaw_center),
            half_range_rotvec=(
                task.recovery_tilt_half_range_rad,
                task.recovery_tilt_half_range_rad,
                yaw_half_range,
            ),
            inward_margin_rotvec=task.rotation_recovery_inward_margin,
            activation_delta_rotvec=task.rotation_recovery_activation_delta,
            max_wrench=task.rotation_recovery_max_wrench,
            wrench_gain_multiplier=task.rotation_recovery_wrench_gain_multiplier,
        )


class Robot(BaseRobot):
    config: RobotConfig

    def connect(self) -> None:
        super().connect()
        task = self.config.task
        target_pose = self.impedance_control.get_actual_tcp_pose()
        yaw = float(np.clip(
            yaw_relative_to_nominal(task, target_pose[1]),
            *task.yaw_range_rad,
        ))
        target_pose[1] = rotation_at_yaw(
            task, yaw
        )
        self.impedance_control.set_target_pose(target_pose)

    def apply_action(self, raw_action: dict) -> None:
        task = self.config.task
        if raw_action.get("gripper_only", False):
            send_continuous_gripper_action(
                self.gripper, float(raw_action["gripper"]),
                speed=task.gripper_speed, force=task.gripper_force,
            )
            return

        actual_pose = self.impedance_control.get_actual_tcp_pose()
        if raw_action.get("hold_current_pose", False):
            # Cancel a stale reset target using robot-side current feedback.
            # The PID's safety recovery remains active when needed.
            self.impedance_control.set_target_pose(actual_pose.copy())
            return
        if self.impedance_control.update_rotation_recovery(actual_pose[1]):
            return

        reset_target_pose = raw_action.get("reset_target_pose")
        if reset_target_pose is not None:
            requested_pose = np.asarray(
                reset_target_pose, dtype=np.float32
            ).reshape(2, 3).copy()
            position_axes = np.asarray(
                raw_action.get("reset_position_axes", (True, True, True)),
                dtype=bool,
            )
            if position_axes.shape != (3,):
                raise ValueError("reset_position_axes must contain XYZ booleans")
            target_pose = actual_pose.copy()
            clipped_position = np.clip(
                requested_pose[0],
                np.asarray(task.safety_pos_min, dtype=np.float32),
                np.asarray(task.safety_pos_max, dtype=np.float32),
            )
            target_pose[0, position_axes] = clipped_position[position_axes]
            if bool(raw_action.get("reset_rotation", True)):
                reset_yaw = float(np.clip(
                    yaw_relative_to_nominal(task, requested_pose[1]),
                    *task.yaw_range_rad,
                ))
                target_pose[1] = self.impedance_control.constrain_rotation_target(
                    rotation_at_yaw(task, reset_yaw)
                )
            self.impedance_control.set_target_pose(target_pose)
            if "gripper" in raw_action:
                send_continuous_gripper_action(
                    self.gripper, float(raw_action["gripper"]),
                    speed=task.gripper_speed, force=task.gripper_force,
                )
            return

        normalized_pos = np.clip(
            np.asarray(
                raw_action.get("delta_pos", np.zeros(3)), dtype=np.float32
            ).reshape(3),
            -1.0,
            1.0,
        )
        target_pose = actual_pose.copy()
        target_pose[0] += normalized_pos * task.linear_speed
        target_pose[0] = np.clip(
            target_pose[0],
            np.asarray(task.safety_pos_min, dtype=np.float32),
            np.asarray(task.safety_pos_max, dtype=np.float32),
        )

        normalized_rot = np.clip(
            np.asarray(
                raw_action.get("delta_rot", np.zeros(3)), dtype=np.float32
            ).reshape(3),
            -1.0,
            1.0,
        )
        previous_target = np.asarray(
            self.impedance_control.target_pose, dtype=np.float32
        ).reshape(2, 3)
        target_yaw = yaw_relative_to_nominal(task, previous_target[1])
        target_yaw += float(normalized_rot[2]) * task.angular_speed
        target_yaw = float(np.clip(target_yaw, *task.yaw_range_rad))
        target_pose[1] = self.impedance_control.constrain_rotation_target(
            rotation_at_yaw(task, target_yaw)
        )
        self.impedance_control.set_target_pose(target_pose)

        if "gripper" in raw_action:
            send_continuous_gripper_action(
                self.gripper, float(raw_action["gripper"]),
                speed=task.gripper_speed, force=task.gripper_force,
            )
