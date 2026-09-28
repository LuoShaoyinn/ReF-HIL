"""Plot validation profiles for saved per-episode actors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def wilson(successes: int, attempts: int) -> tuple[float, float]:
    if attempts < 1:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (rate + z * z / (2.0 * attempts)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / attempts + z * z / (4.0 * attempts**2))
        / denominator
    )
    return center - radius, center + radius


def load_dense(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("actor_episode_*/profile.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        for attempt in profile.get("attempts", []):
            source_episode = attempt.get("source_actor_episode")
            if source_episode is None:
                continue
            rows.append(
                {
                    "actor_episode": int(source_episode),
                    "success": bool(attempt["success"]),
                    "unsafe": attempt.get("outcome") == "unsafe_failure",
                    "steps": int(attempt["scored_steps"]),
                }
            )
    return sorted(rows, key=lambda row: row["actor_episode"])


def load_sparse(experiment: Path) -> list[dict]:
    rows: list[dict] = []
    profile_root = experiment / "validation" / "profiles"
    for path in sorted(profile_root.glob("checkpoint_*/profile.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_name = str(profile["checkpoint"])
        checkpoint_path = experiment / "checkpoints" / checkpoint_name
        actor_episode = -1
        if checkpoint_path.exists():
            with checkpoint_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            actor_episode = int(checkpoint["learner"].get("actor_episodes", -1))
        summary = profile["summary"]
        attempts = int(summary["attempts"])
        successes = int(summary["successes"])
        low, high = wilson(successes, attempts)
        rows.append(
            {
                "checkpoint": checkpoint_name,
                "actor_episode": actor_episode,
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts,
                "low": low,
                "high": high,
            }
        )
    return [row for row in rows if row["actor_episode"] >= 0]


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        result[index] = values[max(0, index - window + 1) : index + 1].mean()
    return result


def aggregate_actors(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["actor_episode"]), []).append(row)
    profiles: list[dict] = []
    for actor_episode, attempts_for_actor in sorted(grouped.items()):
        attempts = len(attempts_for_actor)
        successes = sum(int(row["success"]) for row in attempts_for_actor)
        unsafe_failures = sum(int(row["unsafe"]) for row in attempts_for_actor)
        low, high = wilson(successes, attempts)
        profiles.append(
            {
                "actor_episode": actor_episode,
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts,
                "unsafe_failures": unsafe_failures,
                "unsafe_failure_rate": unsafe_failures / attempts,
                "mean_steps": float(
                    np.mean([row["steps"] for row in attempts_for_actor])
                ),
                "low": low,
                "high": high,
            }
        )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--start-actor-episode", type=int, default=None)
    parser.add_argument("--end-actor-episode", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.rolling_window < 1:
        parser.error("rolling window must be positive")

    experiment = Path("outputs") / args.experiment_name
    dense = load_dense(
        experiment / "episode_actor_validation" / "profiles"
    )
    if args.start_actor_episode is not None:
        dense = [
            row for row in dense
            if row["actor_episode"] >= args.start_actor_episode
        ]
    if args.end_actor_episode is not None:
        dense = [
            row for row in dense
            if row["actor_episode"] <= args.end_actor_episode
        ]
    sparse = load_sparse(experiment)
    if args.start_actor_episode is not None:
        sparse = [
            row for row in sparse
            if row["actor_episode"] >= args.start_actor_episode
        ]
    if args.end_actor_episode is not None:
        sparse = [
            row for row in sparse
            if row["actor_episode"] <= args.end_actor_episode
        ]
    if not dense:
        raise FileNotFoundError(
            f"no per-episode actor profiles in {experiment / 'episode_actor_validation'}"
        )
    output = (
        experiment / "episode_actor_validation_curve.png"
        if args.output is None
        else args.output
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    actor_profiles = aggregate_actors(dense)
    x = np.asarray(
        [row["actor_episode"] for row in actor_profiles], dtype=np.int64
    )
    success_rate = np.asarray(
        [row["success_rate"] for row in actor_profiles], dtype=np.float64
    )
    unsafe_rate = np.asarray(
        [row["unsafe_failure_rate"] for row in actor_profiles], dtype=np.float64
    )
    low = np.asarray([row["low"] for row in actor_profiles], dtype=np.float64)
    high = np.asarray([row["high"] for row in actor_profiles], dtype=np.float64)
    total_attempts = sum(row["attempts"] for row in actor_profiles)
    total_successes = sum(row["successes"] for row in actor_profiles)
    overall_rate = total_successes / total_attempts

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.errorbar(
        x,
        success_rate,
        yerr=np.stack((success_rate - low, high - success_rate)),
        fmt="o-",
        capsize=4,
        linewidth=2.0,
        color="#2d5f9a",
        label="success rate per saved actor (95% Wilson)",
    )
    axis.plot(
        x,
        unsafe_rate,
        marker="x",
        linestyle="--",
        linewidth=1.5,
        color="#d1495b",
        label="unsafe-failure rate",
    )
    axis.axhline(
        overall_rate,
        color="#21a179",
        linewidth=1.5,
        linestyle=":",
        label=(
            f"overall success {total_successes}/{total_attempts} "
            f"({overall_rate:.1%})"
        ),
    )
    if sparse:
        sparse_x = np.asarray([row["actor_episode"] for row in sparse])
        sparse_y = np.asarray([row["success_rate"] for row in sparse])
        low = np.asarray([row["low"] for row in sparse])
        high = np.asarray([row["high"] for row in sparse])
        axis.errorbar(
            sparse_x,
            sparse_y,
            yerr=np.stack((sparse_y - low, high - sparse_y)),
            fmt="o-",
            capsize=4,
            linewidth=1.5,
            color="#f28e2b",
            label="periodic checkpoint profile (95% Wilson)",
        )
    axis.set(
        xlabel="physical training actor episode",
        ylabel="fraction of validation trials",
        ylim=(-0.08, 1.08),
    )
    axis.set_xticks(x)
    axis.set_title(
        f"{args.experiment_name}: end-of-training saved actor validation"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output, dpi=160)
    plt.close(figure)

    data_path = output.with_suffix(".json")
    data_path.write_text(
        json.dumps(
            {
                "attempts": dense,
                "actor_profiles": actor_profiles,
                "overall": {
                    "attempts": total_attempts,
                    "successes": total_successes,
                    "success_rate": overall_rate,
                },
                "checkpoint_profiles": sparse,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
