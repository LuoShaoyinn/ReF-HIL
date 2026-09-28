"""Pure SpaceMouse copilot math without device or ZMQ dependencies."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_copilot_action(
    current_rotvec: np.ndarray,
    target_rotvec: np.ndarray,
    *,
    angular_speed: float,
    gain: float = 1.0,
    deadband_rad: float = 0.01,
) -> np.ndarray:
    """Return normalized local rotation commanding current -> target.

    The task robot post-multiplies local rotation actions and scales them by
    ``angular_speed``, so the matching local error is ``current^-1 * target``.
    """

    if angular_speed <= 0.0:
        raise ValueError("angular_speed must be positive")
    if gain <= 0.0:
        raise ValueError("copilot gain must be positive")
    if deadband_rad < 0.0:
        raise ValueError("copilot deadband must be nonnegative")
    current = R.from_rotvec(np.asarray(current_rotvec, dtype=np.float64).reshape(3))
    target = R.from_rotvec(np.asarray(target_rotvec, dtype=np.float64).reshape(3))
    local_error = (current.inv() * target).as_rotvec()
    if np.linalg.norm(local_error) <= deadband_rad:
        return np.zeros(3, dtype=np.float32)
    return np.clip(gain * local_error / angular_speed, -1.0, 1.0).astype(
        np.float32
    )
