from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True, kw_only=True)
class ResetConfig:
    # Keep a 5 mm margin below the task's 120 mm Cartesian safety ceiling.
    travel_z: float = 0.115
    safety_pos_min: tuple[float, float, float] = (-0.250, -0.650, 0.005)
    safety_pos_max: tuple[float, float, float] = (-0.080, -0.400, 0.120)
    random_xy_margin: float = 0.050
    random_y_strict_min: float = -0.460
    linear_speed: float = 0.050
    move_tolerance: float = 0.005
    move_timeout_s: float = 8.0
    poll_interval_s: float = 0.05
    open_action: float = -1.0


def sample_reset_xy(config: ResetConfig, rng: np.random.Generator) -> np.ndarray:
    lower = np.asarray(config.safety_pos_min[:2], dtype=np.float32) + config.random_xy_margin
    upper = np.asarray(config.safety_pos_max[:2], dtype=np.float32) - config.random_xy_margin
    lower[1] = max(
        lower[1],
        np.nextafter(np.float32(config.random_y_strict_min), np.float32(np.inf)),
    )
    if np.any(lower >= upper):
        raise ValueError(f"empty assemble reset range: lower={lower.tolist()}, upper={upper.tolist()}")
    return rng.uniform(lower, upper).astype(np.float32)


def run_reset(
    send_action: Callable[[dict], None],
    read_observation: Callable[[], dict],
    *,
    config: ResetConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Open, lift vertically, then move to the next random episode start."""

    rng = np.random.default_rng() if rng is None else rng

    def action(position_error: np.ndarray) -> dict:
        return {
            "delta_pos": np.clip(
                np.asarray(position_error, dtype=np.float32) / config.linear_speed,
                -1.0,
                1.0,
            ),
            "delta_rot": np.zeros(3, dtype=np.float32),
        }

    print("reset: open gripper", flush=True)
    open_command = action(np.zeros(3, dtype=np.float32))
    open_command["gripper"] = float(config.open_action)
    send_action(open_command)

    def move_to(step: int, label: str, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32).reshape(3)
        print(f"reset {step}/2: {label} -> {(target * 1000.0).round(1).tolist()} mm", flush=True)
        deadline = time.monotonic() + config.move_timeout_s
        while True:
            current = np.asarray(read_observation()["tcp_pose"], dtype=np.float32).reshape(2, 3)[0]
            error = target - current
            if float(np.linalg.norm(error)) <= config.move_tolerance:
                return
            send_action(action(error))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"assemble reset step {step} ({label}) timed out: "
                    f"target={target.tolist()}, current={current.tolist()}, "
                    f"error={float(np.linalg.norm(error)):.4f} m"
                )
            time.sleep(config.poll_interval_s)

    current = np.asarray(read_observation()["tcp_pose"], dtype=np.float32).reshape(2, 3)[0]
    lift_z = float(
        np.clip(
            max(float(current[2]), config.travel_z),
            config.safety_pos_min[2],
            config.safety_pos_max[2],
        )
    )
    lifted = np.asarray([current[0], current[1], lift_z], dtype=np.float32)
    move_to(1, "lift vertically", lifted)

    next_xy = sample_reset_xy(config, rng)
    next_start = np.asarray([*next_xy, lift_z], dtype=np.float32)
    move_to(2, "move to next episode start", next_start)
    return next_start
