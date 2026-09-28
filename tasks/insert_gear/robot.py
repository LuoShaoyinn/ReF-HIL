"""Gear-insertion robot control: XYZ, local yaw, and fixed closed gripper."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from actor.robot.action import send_continuous_gripper_action
from tasks.gear.robot import Robot as GearRobot
from tasks.gear.robot import RobotConfig as GearRobotConfig
from tasks.gear.modeling import rotation_at_yaw

from .config import TaskConfig


@dataclass(kw_only=True)
class RobotConfig(GearRobotConfig):
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Match rotate_knob: permit RZ to track the 9-degree command increment
        # at the corresponding 90-degree/second physical speed limit.
        self.impendence_control.limit["max_speed"][1, 2] = (
            self.task.yaw_speed_limit_rad_s
        )


class Robot(GearRobot):
    config: RobotConfig

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self._startup_complete = False
        self._startup_thread: threading.Thread | None = None

    def connect(self) -> None:
        super().connect()
        self._startup_thread = threading.Thread(
            target=self._interactive_pickup,
            name="insert-gear-pickup",
            daemon=True,
        )
        self._startup_thread.start()

    def _close_gripper(self) -> None:
        task = self.config.task
        send_continuous_gripper_action(
            self.gripper,
            float(task.fixed_gripper_action),
            speed=task.gripper_speed,
            force=task.gripper_force,
        )

    def apply_action(self, raw_action: dict) -> None:
        # Keep actor/recorder commands from racing the operator-controlled
        # pickup sequence. Observation requests remain serviced by UR5eBase.
        if not self._startup_complete:
            return
        if raw_action.get("insert_gear_reset", False):
            # Explicit reset commands may open the gripper; both reset and
            # ordinary policy/teleop use the configured workspace fence.
            if "reset_target_pose" in raw_action:
                actual = self.impedance_control.get_actual_tcp_pose()
                if self.impedance_control.update_rotation_recovery(actual[1]):
                    return
                task = self.config.task
                target = np.asarray(raw_action["reset_target_pose"], dtype=np.float32).reshape(2, 3).copy()
                if not np.isfinite(target).all():
                    raise ValueError("non-finite insert_gear reset target")
                target[0] = np.clip(
                    target[0], task.safety_pos_min, task.safety_pos_max,
                )
                target[1] = self.impedance_control.constrain_rotation_target(target[1])
                self.impedance_control.set_target_pose(target)
            send_continuous_gripper_action(
                self.gripper, float(raw_action["gripper"]),
                speed=self.config.task.gripper_speed, force=self.config.task.gripper_force,
            )
            return
        if raw_action.get("gripper_only", False):
            self._close_gripper()
            return
        # GearRobot's optional gripper path uses generic defaults. Keep the
        # fixed gripper entirely task-owned so every command uses speed=0 and
        # force=255.
        arm_action = dict(raw_action)
        arm_action.pop("gripper", None)
        super().apply_action(arm_action)
        self._close_gripper()

    def _interactive_pickup(self) -> None:
        task = self.config.task
        try:
            input(
                "[insert_gear] Press Enter to move to the startup pickup pose "
                f"{task.startup_pickup_pos}..."
            )
            pickup_pose = np.asarray(
                [
                    task.startup_pickup_pos,
                    rotation_at_yaw(task, task.startup_pickup_yaw_rad),
                ],
                dtype=np.float32,
            )
            approach_pose = pickup_pose.copy()
            approach_pose[0, 2] = float(task.safety_pos_max[2])
            if not self._set_pose_and_wait(
                approach_pose, timeout_s=task.startup_move_timeout_s
            ):
                self._startup_failed("elevated approach pose")
                return
            if not self._set_pose_and_wait(
                pickup_pose, timeout_s=task.startup_move_timeout_s
            ):
                self._startup_failed("pickup pose")
                return
            input("[insert_gear] Press Enter to close the gripper at maximum force...")
            self._close_gripper()
            input("[insert_gear] Press Enter to lift to the reset height...")
            lift_pose = self.impedance_control.get_actual_tcp_pose()
            lift_pose[0, 2] = float(task.safety_pos_max[2])
            if not self._set_pose_and_wait(
                lift_pose, timeout_s=task.startup_move_timeout_s
            ):
                self._startup_failed("final lifted pose")
                return
            self._startup_complete = True
            print(
                "[insert_gear] Pickup complete; actor/recorder commands are now enabled.",
                flush=True,
            )
        except (EOFError, KeyboardInterrupt):
            print(
                "[insert_gear] Startup pickup was not completed; motion commands remain disabled.",
                flush=True,
            )

    @staticmethod
    def _startup_failed(stage: str) -> None:
        print(
            f"[insert_gear] ERROR: {stage} was not reached; actor/recorder "
            "commands remain disabled. Inspect the robot and restart the driver.",
            flush=True,
        )

    def _set_pose_and_wait(
        self, target_pose: np.ndarray, *, timeout_s: float | None = None
    ) -> bool:
        task = self.config.task
        if timeout_s is None:
            timeout_s = task.reset_stage_timeout_s
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(2, 3).copy()
        target_pose[0] = np.clip(
            target_pose[0],
            np.asarray(task.safety_pos_min, dtype=np.float32),
            np.asarray(task.safety_pos_max, dtype=np.float32),
        )
        target_pose[1] = self.impedance_control.constrain_rotation_target(target_pose[1])
        self.impedance_control.set_target_pose(target_pose)
        started = time.monotonic()
        while time.monotonic() - started < timeout_s:
            current = self.impedance_control.get_actual_tcp_pose()
            position_error = np.linalg.norm(target_pose[0] - current[0])
            rotation_error = np.linalg.norm(
                (
                    Rotation.from_rotvec(current[1]).inv()
                    * Rotation.from_rotvec(target_pose[1])
                ).as_rotvec()
            )
            if (
                position_error <= task.reset_position_tolerance
                and rotation_error <= task.reset_rotation_tolerance
            ):
                return True
            time.sleep(task.reset_step_duration_s)
        return False
