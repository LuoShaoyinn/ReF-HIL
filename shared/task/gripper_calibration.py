"""Invert the robot's task-specific gripper action mapping for teleoperation."""

from collections.abc import Mapping
import numpy as np


def gripper_control_position(
    observation: Mapping,
    fallback_range: tuple[float, float],
) -> float:
    """Return [0=open, 1=closed] using the current driver's effective endpoints.

    Old drivers/recordings lack metadata and retain their task-config fallback.
    Present but invalid metadata is an error, never a reason to command open.
    This is control feedback, not a change to saved policy proprioception.
    """
    endpoints = np.asarray(
        observation.get("gripper_action_range", fallback_range), dtype=np.float64
    )
    if endpoints.shape != (2,) or not np.isfinite(endpoints).all():
        raise ValueError("gripper_action_range must contain two finite endpoints")
    opened, closed = map(float, endpoints)
    if not opened < closed:
        raise ValueError("gripper_action_range requires open < closed")
    position = float(np.asarray(observation["gripper"]).reshape(()))
    if not np.isfinite(position):
        raise ValueError("gripper position must be finite")
    return float(np.clip((position - opened) / (closed - opened), 0.0, 1.0))
