from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def apply_local_rotation_delta(
    actual_rotvec: np.ndarray,
    delta_rotvec: np.ndarray,
) -> np.ndarray:
    """Post-multiply a gripper-local rotation delta onto the TCP orientation."""

    actual = R.from_rotvec(np.asarray(actual_rotvec, dtype=np.float64).reshape(3))
    delta = R.from_rotvec(np.asarray(delta_rotvec, dtype=np.float64).reshape(3))
    return (actual * delta).as_rotvec()


def clip_rotation_relative_to_nominal(
    candidate_rotvec: np.ndarray,
    nominal_rotvec: np.ndarray,
    half_range_rotvec: np.ndarray,
) -> np.ndarray:
    """Clip orientation in the nominal gripper frame, away from the pi branch."""

    candidate = R.from_rotvec(
        np.asarray(candidate_rotvec, dtype=np.float64).reshape(3)
    )
    nominal = R.from_rotvec(
        np.asarray(nominal_rotvec, dtype=np.float64).reshape(3)
    )
    half_range = np.asarray(half_range_rotvec, dtype=np.float64).reshape(3)
    relative = (nominal.inv() * candidate).as_rotvec()
    clipped_relative = np.clip(relative, -half_range, half_range)
    return (nominal * R.from_rotvec(clipped_relative)).as_rotvec()
