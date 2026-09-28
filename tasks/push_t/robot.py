"""Push-T robot control: XYZ translation and fixed orientation/gripper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from actor.robot.action import send_continuous_gripper_action
from shared.task.manipulation_robot import Robot as BaseRobot
from shared.task.manipulation_robot import RobotConfig as BaseRobotConfig
from .config import TaskConfig


@dataclass(kw_only=True)
class RobotConfig(BaseRobotConfig):
    task: TaskConfig = field(default_factory=TaskConfig)


class Robot(BaseRobot):
    config: RobotConfig

    def _close_gripper(self) -> None:
        send_continuous_gripper_action(
            self.gripper, float(self.config.task.fixed_gripper_action)
        )

    def connect(self) -> None:
        super().connect()
        task = self.config.task
        target_pose = self.impedance_control.get_actual_tcp_pose()
        # Hold the measured Cartesian position at startup while immediately
        # establishing the task's fixed canonical orientation target.
        target_pose[1] = np.asarray(task.zero_point_rot, dtype=np.float32)
        self.impedance_control.set_target_pose(target_pose)
        self._close_gripper()

    def apply_action(self, raw_action: dict) -> None:
        task = self.config.task
        actual_pose = self.impedance_control.get_actual_tcp_pose()

        reset_target_pose = raw_action.get("reset_target_pose")
        if reset_target_pose is not None:
            target_pose = np.asarray(
                reset_target_pose, dtype=np.float32
            ).reshape(2, 3).copy()
            target_pose[0] = np.clip(
                target_pose[0],
                np.asarray(task.safety_pos_min, dtype=np.float32),
                np.asarray(task.safety_pos_max, dtype=np.float32),
            )
            target_pose[1] = np.asarray(task.zero_point_rot, dtype=np.float32)
            self.impedance_control.set_target_pose(target_pose)
            self._close_gripper()
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
        target_pose[1] = np.asarray(task.zero_point_rot, dtype=np.float32)
        self.impedance_control.set_target_pose(target_pose)
        self._close_gripper()
