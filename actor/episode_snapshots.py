"""Exact policy snapshots used by individual physical actor episodes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pickle


EPISODE_ACTOR_FORMAT = "physical_episode_actor_v1"


def episode_actor_index(path: Path) -> int:
    suffix = path.stem.removeprefix("actor_episode_")
    if not suffix.isdigit():
        raise ValueError(f"invalid episode actor filename: {path.name}")
    return int(suffix)


def list_episode_actors(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("actor_episode_*.pkl"),
        key=episode_actor_index,
    )


def next_episode_actor_index(directory: Path) -> int:
    paths = list_episode_actors(directory)
    return 1 if not paths else episode_actor_index(paths[-1]) + 1


def episode_actors_from(paths: list[Path], start: int | None) -> list[Path]:
    if start is None:
        return paths
    if start < 1:
        raise ValueError("start actor episode must be positive")
    if not any(episode_actor_index(path) == start for path in paths):
        raise FileNotFoundError(f"actor episode {start} was not found")
    return [path for path in paths if episode_actor_index(path) >= start]


def episode_actor_named(directory: Path, name: str) -> Path:
    """Resolve exactly one actor snapshot by basename or numeric episode id."""

    token = Path(name).name
    if token != name:
        raise ValueError("actor name must be a filename or numeric episode id")
    if token.isdigit():
        token = f"actor_episode_{int(token):06d}.pkl"
    path = directory / token
    if path.name != token or not path.is_file():
        raise FileNotFoundError(f"episode actor was not found: {path}")
    episode_actor_index(path)
    return path


def write_episode_actor(
    *,
    directory: Path,
    episode: int,
    policy_revision: int,
    actor_snapshot: dict,
) -> Path:
    """Atomically save the exact inference snapshot used by one episode."""

    if episode < 1:
        raise ValueError("episode must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"actor_episode_{episode:06d}.pkl"
    temporary = path.with_suffix(".pkl.tmp")
    runtime = actor_snapshot.get("runtime", {})
    learner = {
        "train_steps": int(runtime.get("train_steps", -1)),
        "actor_steps": int(runtime.get("actor_steps", -1)),
        "actor_episodes": int(runtime.get("actor_episodes", -1)),
    }
    payload = {
        "format_version": 1,
        "class_name": "PhysicalEpisodeActor",
        "episode_actor_format": EPISODE_ACTOR_FORMAT,
        "actor_episode": int(episode),
        "policy_revision": int(policy_revision),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": int(runtime.get("elapsed_seconds", -1)),
        "learner": learner,
        "actor": actor_snapshot,
    }
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
    temporary.replace(path)
    return path
