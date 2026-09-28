"""Calibrated paper double-string task with two success conditions."""

from __future__ import annotations

from dataclasses import dataclass

from shared.task.manipulation_config import (
    SuccessCondition,
    TaskConfig as BaseTaskConfig,
)


@dataclass(frozen=True, kw_only=True)
class TaskConfig(BaseTaskConfig):
    name: str = "hang_double_strings_2"
    max_episode_steps: int = 200
    human_string_reset_seconds: int = 5
    gripper_speed: int = 128
    gripper_force: int = 0

    # Calibrated Cartesian workspace, in metres.
    safety_pos_min: tuple[float, float, float] = (-0.320, -0.700, -0.004)
    safety_pos_max: tuple[float, float, float] = (-0.200, -0.450, 0.060)

    # Reset at fixed X/Z and sample Y uniformly from [-0.640, -0.560] m.
    reset_pos: tuple[float, float, float] = (-0.300, -0.600, 0.060)
    reset_rnd_abs: tuple[float, float, float] = (0.0, 0.040, 0.0)

    # Global crop coordinates use the native 1280x720 camera frame.
    global_camera_resolution: tuple[int, int] | None = (1280, 720)
    global_policy_clip: tuple[int, int, int, int] = (350, 94, 448, 448)

    # Match the current rotate_knob wrist-camera views.
    wrist_0_policy_clip: tuple[int, int, int, int] = (0, 0, 448, 448)
    wrist_1_policy_clip: tuple[int, int, int, int] = (0, 0, 448, 448)

    # Raw global-camera crop format: (x, y, width, height). These regions are
    # the two classifier crops supplied for the current setup.
    # Task success is the logical AND of the two independently trained models.
    success_conditions: tuple[SuccessCondition, ...] = (
        SuccessCondition(
            name="upper_clip",
            clip=(440, 328, 80, 80),
            image_key="classifier_upper_clip",
            depth_key="classifier_upper_clip_depth",
            checkpoint="classifiers/upper_clip.pt",
        ),
        SuccessCondition(
            name="lower_clip",
            clip=(615, 201, 64, 64),
            image_key="classifier_lower_clip",
            depth_key="classifier_lower_clip_depth",
            checkpoint="classifiers/lower_clip.pt",
        ),
    )
