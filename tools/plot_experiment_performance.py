"""Plot episode-level performance for a completed SRT experiment."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_episodes(source: Path, copied: Path) -> list[dict]:
    copied.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []
    pending: list[tuple[bool, float, bool]] = []
    for original in sorted(source.glob("*.pkl")):
        snapshot = copied / original.name
        subprocess.run(
            ["cp", "--reflink=auto", str(original), str(snapshot)], check=True
        )
        with snapshot.open("rb") as stream:
            transitions = pickle.load(stream)
        for transition in transitions:
            info = transition["info"]
            pending.append(
                (
                    bool(info.get("is_intervene")),
                    float(transition["reward"]),
                    bool(info.get("forced_success")),
                )
            )
            if not transition["done"]:
                continue
            human_steps = sum(row[0] for row in pending)
            success = pending[-1][1] > 0.0
            forced = pending[-1][2]
            episodes.append(
                {
                    "steps": len(pending),
                    "human_steps": human_steps,
                    "success": success,
                    "forced_success": forced,
                    "auto_success": success and human_steps == 0 and not forced,
                }
            )
            pending = []
    if pending:
        print(f"warning: ignoring {len(pending)} incomplete trailing transitions")
    return episodes


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        [np.mean(values[i - window + 1 : i + 1]) if i >= window - 1 else np.nan
         for i in range(len(values))]
    )


def concatenated_episode_minutes(tb_directory: Path, tag: str) -> np.ndarray:
    """Join TensorBoard process segments without counting restart downtime."""

    elapsed: list[float] = []
    cursor = 0.0
    for event_file in sorted(tb_directory.glob("events.out.tfevents*")):
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        origin = float(events[0].wall_time)
        segment = [(float(event.wall_time) - origin) / 60.0 for event in events]
        elapsed.extend(cursor + value for value in segment)
        cursor += segment[-1]
    return np.asarray(elapsed, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--demos", type=int, default=20)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--concatenate-tensorboard-gaps",
        action="store_true",
        help="remove downtime between TensorBoard process/restart files",
    )
    args = parser.parse_args()

    run = Path("outputs") / args.run
    output = Path("outputs") / f"analysis_{args.run}_performance"
    output.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(run / "buffer", output / "copied_buffer")
    demos, online = episodes[: args.demos], episodes[args.demos :]
    if len(online) < args.window:
        raise ValueError(f"need at least {args.window} online episodes; found {len(online)}")

    cumulative_steps = np.cumsum([row["steps"] for row in online])
    x = np.arange(1, len(online) + 1, dtype=float)
    x_label = "Online episode"
    event_files = list((run / "tb").glob("events.out.tfevents*"))
    if event_files:
        copied_tb = output / "copied_tb"
        shutil.copytree(run / "tb", copied_tb, dirs_exist_ok=True)
        accumulator = EventAccumulator(str(copied_tb), size_guidance={"scalars": 0})
        accumulator.Reload()
        events = accumulator.Scalars("Actor/intervention_fraction")
        if args.concatenate_tensorboard_gaps:
            concatenated = concatenated_episode_minutes(
                run / "tb", "Actor/intervention_fraction"
            )
            if len(concatenated) != len(online):
                raise ValueError(
                    "TensorBoard episode count does not match replay: "
                    f"{len(concatenated)} versus {len(online)}"
                )
            x = concatenated
            x_label = "Active training minutes (restart gaps removed)"
        else:
            by_step = {event.step: event for event in events}
            if all(int(step) in by_step for step in cumulative_steps):
                start = min(
                    event.wall_time
                    for tag in accumulator.Tags()["scalars"]
                    for event in accumulator.Scalars(tag)
                )
                x = np.asarray(
                    [(by_step[int(step)].wall_time - start) / 60.0 for step in cumulative_steps]
                )
                x_label = "Minutes since training log start"

    success = np.asarray([row["success"] for row in online], dtype=float)
    autonomous = np.asarray([row["auto_success"] for row in online], dtype=float)
    lengths = np.asarray([row["steps"] for row in online], dtype=float)
    intervention = np.asarray(
        [row["human_steps"] / row["steps"] for row in online], dtype=float
    )
    autonomous_length = np.asarray(
        [
            np.mean(lengths[i - args.window + 1 : i + 1][
                autonomous[i - args.window + 1 : i + 1].astype(bool)
            ])
            if i >= args.window - 1
            and autonomous[i - args.window + 1 : i + 1].any()
            else np.nan
            for i in range(len(online))
        ]
    )
    demo_lengths = np.asarray([row["steps"] for row in demos if row["success"]])

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    figure.suptitle(
        f"{args.run} | trailing {args.window} online episodes\n"
        "Intervened or forced-success episodes are not autonomous successes"
    )
    axes[0].plot(x, 100 * rolling(autonomous, args.window), label="Autonomous success")
    axes[0].plot(x, 100 * rolling(success, args.window), alpha=0.65, label="Any success")
    axes[0].set(ylabel="Success rate (%)", ylim=(-3, 103))

    mask = autonomous.astype(bool)
    axes[1].scatter(x[mask], lengths[mask], s=14, alpha=0.45, label="Autonomous success")
    axes[1].plot(x, autonomous_length, label="Window mean autonomous length")
    for reducer, label, color in (
        (np.min, "Demo minimum", "tab:green"),
        (np.mean, "Demo mean", "tab:pink"),
        (np.max, "Demo maximum", "tab:purple"),
    ):
        value = float(reducer(demo_lengths))
        axes[1].axhline(value, linestyle=":", color=color, label=f"{label}: {value:.1f}")
    axes[1].set(ylabel="Episode length (steps)")

    axes[2].plot(
        x,
        100 * rolling(intervention, args.window),
        color="tab:orange",
        label="Mean intervention-step fraction",
    )
    axes[2].set(ylabel="Intervention steps (%)", xlabel=x_label, ylim=(-3, 103))
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, loc="best")
    figure.savefig(output / "performance.png", dpi=180)

    summary = {
        "demo_episodes": len(demos),
        "online_episodes": len(online),
        "window": args.window,
        "overall_autonomous_success_rate": float(np.mean(autonomous)),
        "tail_autonomous_success_rate": float(np.mean(autonomous[-args.window :])),
        "best_window_autonomous_success_rate": float(
            np.nanmax(rolling(autonomous, args.window))
        ),
        "tail_any_success_rate": float(np.mean(success[-args.window :])),
        "demo_steps": {
            "minimum": int(np.min(demo_lengths)),
            "mean": float(np.mean(demo_lengths)),
            "maximum": int(np.max(demo_lengths)),
        },
        "x_axis": x_label,
        "end_x": float(x[-1]),
    }
    (output / "episodes.json").write_text(json.dumps(episodes, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(output / "performance.png")


if __name__ == "__main__":
    main()
