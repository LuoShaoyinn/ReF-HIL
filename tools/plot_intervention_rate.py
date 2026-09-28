"""Plot per-episode and rolling intervention rates from streamed replay chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_episodes(buffer_dir: Path) -> list[dict]:
    episodes: list[dict] = []
    current_steps = 0
    current_interventions = 0
    expected_step = 1

    for path in sorted(buffer_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        for row in rows:
            info = row.get("info", {})
            step = int(info.get("steps_in_episode", expected_step))
            if step == 1 and current_steps:
                episodes.append(
                    {
                        "steps": current_steps,
                        "intervention_steps": current_interventions,
                    }
                )
                current_steps = 0
                current_interventions = 0
            elif step != expected_step:
                raise ValueError(
                    f"non-contiguous episode at {path}: expected step "
                    f"{expected_step}, got {step}"
                )

            current_steps += 1
            current_interventions += int(bool(info.get("is_intervene", False)))
            expected_step = step + 1
            if bool(row.get("done", False)):
                episodes.append(
                    {
                        "steps": current_steps,
                        "intervention_steps": current_interventions,
                    }
                )
                current_steps = 0
                current_interventions = 0
                expected_step = 1

    if current_steps:
        episodes.append(
            {
                "steps": current_steps,
                "intervention_steps": current_interventions,
                "partial": True,
            }
        )
    return episodes


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = values[start : index + 1].mean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--demo-episodes", type=int, default=20)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--validation-start", type=int, default=None)
    parser.add_argument("--validation-end", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.demo_episodes < 0 or args.window < 1:
        parser.error("demo episodes must be nonnegative and window must be positive")

    experiment = Path("outputs") / args.experiment_name
    all_episodes = load_episodes(experiment / "buffer")
    if len(all_episodes) <= args.demo_episodes:
        raise RuntimeError(
            f"found only {len(all_episodes)} episodes, not enough to exclude "
            f"{args.demo_episodes} demonstrations"
        )
    training = all_episodes[args.demo_episodes :]
    for actor_episode, row in enumerate(training, start=1):
        row["actor_episode"] = actor_episode
        row["intervention_rate"] = (
            row["intervention_steps"] / row["steps"]
        )

    x = np.asarray([row["actor_episode"] for row in training], dtype=np.int64)
    rates = np.asarray([row["intervention_rate"] for row in training])
    smoothed = rolling_mean(rates, args.window)
    output = args.output or experiment / "intervention_rate.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.plot(
        x,
        rates,
        color="#9aa0a6",
        linewidth=0.8,
        alpha=0.55,
        label="per-episode intervention rate",
    )
    axis.plot(
        x,
        smoothed,
        color="#d1495b",
        linewidth=2.2,
        label=f"trailing mean ({args.window} episodes)",
    )
    if args.validation_start is not None:
        validation_end = args.validation_end or int(x[-1])
        axis.axvspan(
            args.validation_start - 0.5,
            validation_end + 0.5,
            color="#2d5f9a",
            alpha=0.12,
            label="validated saved actors",
        )
    axis.set(
        xlabel="learner-received online episode",
        ylabel="intervention steps / episode steps",
        ylim=(-0.03, 1.03),
        title=f"{args.experiment_name}: human intervention rate",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output, dpi=160)
    plt.close(figure)

    data = {
        "experiment": args.experiment_name,
        "excluded_demo_episodes": args.demo_episodes,
        "window": args.window,
        "episodes": training,
    }
    output.with_suffix(".json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.json')}")
    print(
        f"training episodes={len(training)}; "
        f"overall intervention rate={sum(row['intervention_steps'] for row in training) / sum(row['steps'] for row in training):.4f}; "
        f"final-window mean={smoothed[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
