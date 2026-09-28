"""The versioned actor/learner wire and raw-data contract.

Only this module may be imported by both runtimes for transport concerns.
Raw transitions deliberately remain plain pickle-compatible dictionaries.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np


PROTOCOL_VERSION = 3
DictMessage: TypeAlias = dict[str, Any]

RAW_TRANSITION_FIELDS = frozenset({"raw_obs", "raw_action", "reward", "done", "info"})


def _normalized_action_values(action: DictMessage, *, label: str) -> np.ndarray:
    if not isinstance(action, dict):
        raise TypeError(f"{label} must be a dict")
    values = np.concatenate(
        (
            np.asarray(action.get("delta_pos"), dtype=np.float32).reshape(3),
            np.asarray(
                action.get("delta_rot", np.zeros(3)), dtype=np.float32
            ).reshape(3),
            np.asarray([action.get("gripper")], dtype=np.float32),
        )
    )
    if not np.isfinite(values).all() or bool(
        (np.abs(values) > 1.0 + 1e-6).any()
    ):
        raise ValueError(f"every {label} component must lie in [-1, 1]")
    return values


def validate_transition(transition: DictMessage) -> None:
    missing = RAW_TRANSITION_FIELDS.difference(transition)
    if missing:
        raise ValueError(f"raw transition is missing fields: {sorted(missing)}")
    if not isinstance(transition["raw_obs"], dict):
        raise TypeError("raw_obs must be a dict")
    if not isinstance(transition["raw_action"], dict):
        raise TypeError("raw_action must be a dict")
    if not isinstance(transition["info"], dict):
        raise TypeError("info must be a dict")
    if transition["info"].get("forced_failure", False):
        if not transition["done"] or float(transition["reward"]) > 0.0:
            raise ValueError("forced_failure must end collection with a nonpositive reward")
        if not isinstance(transition.get("raw_next_obs"), dict):
            raise ValueError(
                "forced_failure must preserve the observed failed next state"
            )
        if transition["info"].get("forced_success", False):
            raise ValueError("a transition cannot be forced success and failure")
    observation = transition["raw_obs"]
    if "tcp_pose" in observation:
        raise ValueError("protocol v3 observations must not contain tcp_pose")
    for key, size in (("tcp_speed", 6), ("tcp_force", 6)):
        value = np.asarray(observation.get(key), dtype=np.float32).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite normalized {size}D vector")
        if bool((np.abs(value) > 1.0 + 1e-6).any()):
            raise ValueError(f"{key} must lie in [-1, 1]")
    gripper_observation = float(np.asarray(observation.get("gripper")).reshape(()))
    if not np.isfinite(gripper_observation) or not -1.0 <= gripper_observation <= 1.0:
        raise ValueError("observed gripper must lie in [-1, 1]")
    projected_gravity = np.asarray(
        observation.get("projected_gravity"), dtype=np.float32
    ).reshape(-1)
    if projected_gravity.shape != (3,) or not np.isfinite(projected_gravity).all():
        raise ValueError("projected_gravity must be a finite 3D vector")
    if bool((np.abs(projected_gravity) > 1.0 + 1e-6).any()):
        raise ValueError("projected_gravity must lie in [-1, 1]")
    if not np.isclose(np.linalg.norm(projected_gravity), 1.0, atol=1e-4):
        raise ValueError("projected_gravity must be a unit vector")
    _normalized_action_values(transition["raw_action"], label="action")
    autonomous_action = transition["info"].get("autonomous_action")
    if autonomous_action is not None:
        if not bool(transition["info"].get("is_intervene", False)):
            raise ValueError(
                "autonomous_action is only valid on an intervened transition"
            )
        _normalized_action_values(autonomous_action, label="autonomous action")


def validate_episode(episode: list[DictMessage]) -> None:
    if not episode:
        raise ValueError("episode must contain at least one transition")
    for transition in episode:
        validate_transition(transition)
    if not bool(episode[-1]["done"]):
        raise ValueError("episode must end in a terminal transition")
