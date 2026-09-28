"""Stateless SpaceMouse gripper-button mapping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def spacemouse_gripper_buttons(
    buttons: Sequence[int | bool],
) -> tuple[bool, bool]:
    """Return ``(close, open)`` for the physical right/left buttons."""

    return bool(buttons[1]), bool(buttons[0])


def resolve_gripper_command(
    buttons: Sequence[int | bool],
    current_gripper: float | None,
    *,
    fallback: float = -1.0,
    last_command: float | None = None,
    hold_tolerance: float = 0.02,
) -> float:
    """Resolve an absolute actor-space gripper command in ``[-1, 1]``.

    Left/open wins if both buttons are pressed. With neither button pressed,
    hold the last command when readback is within ``hold_tolerance`` in
    robot position units (0=open, 1=closed); otherwise follow readback.
    ``fallback`` is used only when that measurement is unavailable.
    """

    close_pressed, open_pressed = spacemouse_gripper_buttons(buttons)
    if open_pressed:
        return -1.0
    if close_pressed:
        return 1.0
    if current_gripper is None:
        return max(-1.0, min(1.0, float(fallback)))
    normalized_position = max(0.0, min(1.0, float(current_gripper)))
    if last_command is not None:
        previous = max(-1.0, min(1.0, float(last_command)))
        if abs(normalized_position - (previous + 1.0) / 2.0) <= hold_tolerance:
            return previous
    return 2.0 * normalized_position - 1.0


def spacemouse_has_intent(
    delta_pos: Sequence[float],
    delta_rot: Sequence[float],
    buttons: Sequence[int | bool],
    *,
    motion_threshold: float = 1e-3,
) -> bool:
    """Return whether any manual axis or either gripper button is active."""

    close_pressed, open_pressed = spacemouse_gripper_buttons(buttons)
    return bool(
        np.linalg.norm(np.asarray(delta_pos, dtype=np.float32).reshape(3))
        > motion_threshold
        or np.linalg.norm(np.asarray(delta_rot, dtype=np.float32).reshape(3))
        > motion_threshold
        or close_pressed
        or open_pressed
    )


__all__ = [
    "resolve_gripper_command",
    "spacemouse_gripper_buttons",
    "spacemouse_has_intent",
]
