"""List saved episode actors whose physical training episode succeeded."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from actor.episode_snapshots import list_episode_actors


@dataclass(frozen=True)
class TrainingEpisode:
    actor_path: Path
    success: bool
    steps: int
    wall_time: float


def format_elapsed(seconds: float) -> str:
    if seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_training_episodes(run: Path) -> list[TrainingEpisode]:
    actor_paths = list_episode_actors(run / "episode_actors")
    if not actor_paths:
        raise FileNotFoundError(f"no episode actors in {run / 'episode_actors'}")

    accumulator = EventAccumulator(
        str(run / "tb"), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if "Actor/success" not in scalar_tags:
        raise FileNotFoundError(f"no Actor/success TensorBoard log in {run / 'tb'}")
    success_events = accumulator.Scalars("Actor/success")
    if len(actor_paths) < len(success_events):
        raise ValueError(
            f"only {len(actor_paths)} episode actors for "
            f"{len(success_events)} completed training episodes"
        )

    # Snapshots are written immediately before physical episodes. A trailing
    # unmatched snapshot is expected when the actor is interrupted.
    expected_names = [
        f"actor_episode_{index:06d}.pkl"
        for index in range(1, len(actor_paths) + 1)
    ]
    if [path.name for path in actor_paths] != expected_names:
        raise ValueError("episode actor files are not contiguous from episode 1")

    episodes: list[TrainingEpisode] = []
    previous_cumulative_steps = 0
    for index, event in enumerate(success_events):
        cumulative_steps = int(event.step)
        steps = cumulative_steps - previous_cumulative_steps
        if steps <= 0:
            raise ValueError("Actor/success steps are not strictly increasing")
        previous_cumulative_steps = cumulative_steps
        episodes.append(
            TrainingEpisode(
                actor_path=actor_paths[index],
                success=float(event.value) > 0.5,
                steps=steps,
                wall_time=float(event.wall_time),
            )
        )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()

    run = Path("outputs") / args.experiment_name
    if not run.is_dir():
        raise FileNotFoundError(
            f"experiment not found: {run}; check the exact directory name"
        )
    episodes = load_training_episodes(run)
    successful = [episode for episode in episodes if episode.success]
    if not successful:
        print("No successful physical training episodes found.")
        return

    first_wall_time = episodes[0].wall_time
    print("actor_snapshot  learner_time  run_time  success_steps")
    for episode in successful:
        with episode.actor_path.open("rb") as handle:
            actor = pickle.load(handle)
        print(
            f"{episode.actor_path.name}  "
            f"{format_elapsed(float(actor.get('elapsed_seconds', -1)))}  "
            f"{format_elapsed(episode.wall_time - first_wall_time)}  "
            f"{episode.steps}"
        )
    print(
        f"Successful training episodes: {len(successful)}/{len(episodes)}; "
        f"saved actors: {len(list_episode_actors(run / 'episode_actors'))}."
    )


if __name__ == "__main__":
    main()
