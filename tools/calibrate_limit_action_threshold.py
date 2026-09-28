"""Calibrate the limit-action radius from held-out initial demonstrations.

This is an offline analysis tool. It never edits policy configuration,
checkpoints, replay, or the online learner.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch

from learner.action_proximity import ACTION_LIKENESS_SIGMA
from learner.action_radius_calibration import (
    EncodedTeachingData,
    episode_folds,
    error_summary,
    prediction_errors,
    radius_to_h_score,
    run_episode_cross_validation,
    train_bc_actor,
)
from tasks import load_component


PREVIOUS_ACTION_LIKENESS_RADIUS = 0.35


QUANTILES = (0.90, 0.95, 0.975)


def load_initial_teaching_episodes(
    buffer_dir: Path, *, episode_count: int
) -> list[list[dict]]:
    """Load exactly the first complete successful fully-human episodes."""

    episodes: list[list[dict]] = []
    current: list[dict] = []
    for path in sorted(buffer_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"replay chunk is not a list: {path}")
        for row in payload:
            current.append(row)
            if not bool(row.get("done", False)):
                continue
            episode_id = len(episodes)
            if float(row.get("reward", 0.0)) <= 0.0:
                raise RuntimeError(
                    f"initial teaching episode {episode_id} did not succeed"
                )
            if not all(
                bool(step.get("info", {}).get("is_intervene", False))
                for step in current
            ):
                raise RuntimeError(
                    f"initial teaching episode {episode_id} is not fully human"
                )
            episodes.append(current)
            current = []
            if len(episodes) == episode_count:
                return episodes
    raise RuntimeError(
        f"need {episode_count} complete successful teaching episodes in "
        f"{buffer_dir}, found {len(episodes)}"
    )


def encode_episodes(
    task: object,
    episodes: list[list[dict]],
    *,
    encode_batch_size: int,
) -> EncodedTeachingData:
    rows = [row for episode in episodes for row in episode]
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for start in range(0, len(rows), encode_batch_size):
        batch = rows[start : start + encode_batch_size]
        observations.append(
            task.build_observations(
                [row["raw_obs"] for row in batch],
                [row["info"] for row in batch],
                augment=False,
            ).detach().cpu().float()
        )
        actions.append(
            task.build_actions(
                [row["raw_action"] for row in batch],
                [row["info"] for row in batch],
                augment=False,
            ).detach().cpu().float()
        )
    episode_ids = torch.cat(
        [torch.full((len(episode),), index) for index, episode in enumerate(episodes)]
    ).long()
    step_ids = torch.cat(
        [torch.arange(len(episode)) for episode in episodes]
    ).long()
    observation = torch.cat(observations)
    action = torch.cat(actions)
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("encoded observations contain non-finite values")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("encoded actions contain non-finite values")
    if bool((action.abs() > 1.00001).any()):
        raise ValueError("teaching actions are not normalized to [-1,1]")
    return EncodedTeachingData(
        observations=observation,
        actions=action,
        episode_ids=episode_ids,
        step_ids=step_ids,
        episode_lengths=tuple(len(episode) for episode in episodes),
    )


def run_leave_one_episode_out(
    data: EncodedTeachingData,
    *,
    model_seed: int,
    mechanism_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    encoder_dim: int,
    hidden_dim: int,
) -> tuple[np.ndarray, list[dict]]:
    errors = np.full(len(data.actions), np.nan, dtype=np.float64)
    reports: list[dict] = []
    all_episodes = set(range(len(data.episode_lengths)))
    for held_out in range(len(data.episode_lengths)):
        training = sorted(all_episodes.difference((held_out,)))
        print(
            f"[LOEO recovery actor] episode {held_out + 1}/"
            f"{len(data.episode_lengths)}",
            flush=True,
        )
        model, training_loss = train_bc_actor(
            data,
            training,
            mechanism_dim=mechanism_dim,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            seed=model_seed + held_out,
        )
        indices, held_out_error = prediction_errors(
            model, data, [held_out], device=device, batch_size=batch_size
        )
        values = held_out_error.numpy().astype(np.float64)
        errors[indices.numpy()] = values
        reports.append(
            {
                "held_out_episode_id": held_out,
                "training_episode_ids": training,
                "final_training_mse": training_loss,
                "held_out_error": error_summary(values),
            }
        )
        del model
    if not np.isfinite(errors).all():
        raise AssertionError("leave-one-episode-out did not evaluate every row")
    return errors, reports


@torch.inference_mode()
def cross_episode_nearest_neighbor_disagreement(
    data: EncodedTeachingData,
    *,
    device: torch.device,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest standardized observation only in another episode."""

    observation = data.observations.float()
    mean = observation.mean(dim=0)
    std = observation.std(dim=0)
    active = std > 1e-4
    standardized = torch.zeros_like(observation)
    standardized[:, active] = (
        observation[:, active] - mean[active]
    ) / std[active]
    reference = standardized.to(device, non_blocking=True)
    reference_episode = data.episode_ids.to(device)
    nearest_indices: list[torch.Tensor] = []
    nearest_distances: list[torch.Tensor] = []
    scale = math.sqrt(max(1, int(active.sum())))
    for start in range(0, len(observation), query_batch_size):
        end = min(len(observation), start + query_batch_size)
        query = reference[start:end]
        distance = torch.cdist(query, reference) / scale
        same_episode = (
            data.episode_ids[start:end].to(device)[:, None]
            == reference_episode[None, :]
        )
        distance.masked_fill_(same_episode, float("inf"))
        value, index = distance.min(dim=1)
        nearest_indices.append(index.cpu())
        nearest_distances.append(value.cpu())
    nearest = torch.cat(nearest_indices)
    state_distance = torch.cat(nearest_distances)
    disagreement = torch.linalg.vector_norm(
        data.actions - data.actions[nearest], dim=1
    )
    if bool(data.episode_ids.eq(data.episode_ids[nearest]).any()):
        raise AssertionError("nearest-neighbor diagnostic leaked within episode")
    return (
        nearest.numpy().astype(np.int64),
        state_distance.numpy().astype(np.float64),
        disagreement.numpy().astype(np.float64),
    )


def candidate_thresholds(errors: np.ndarray, *, sigma: float) -> dict[str, dict]:
    labels = {0.90: "q90", 0.95: "q95", 0.975: "q97_5"}
    return {
        labels[quantile]: {
            "quantile": quantile,
            "radius": float(np.quantile(errors, quantile)),
            "held_out_coverage": float(
                np.mean(errors <= float(np.quantile(errors, quantile)))
            ),
            "h_score_threshold": radius_to_h_score(
                float(np.quantile(errors, quantile)), sigma=sigma
            ),
        }
        for quantile in QUANTILES
    }


def _cdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def draw_empirical_cdf(
    output: Path,
    *,
    cv_error: np.ndarray,
    loeo_error: np.ndarray,
    nn_disagreement: np.ndarray,
    candidates: dict[str, dict],
) -> None:
    """Draw dependency-free CDF diagnostics using the existing OpenCV stack."""

    width, height = 1500, 950
    left, right, top, bottom = 115, 50, 80, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    all_values = np.concatenate((cv_error, loeo_error, nn_disagreement))
    x_max = max(0.05, float(np.quantile(all_values, 0.995)) * 1.08)
    colors = {
        "5-fold held-out BC error": (40, 90, 220),
        "LOEO recovery-actor error": (30, 150, 60),
        "cross-episode NN action disagreement": (200, 100, 20),
    }
    for fraction in np.linspace(0.0, 1.0, 6):
        y = top + plot_height - int(round(fraction * plot_height))
        cv2.line(canvas, (left, y), (left + plot_width, y), (220, 220, 220), 1)
        cv2.putText(
            canvas, f"{fraction:.1f}", (48, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA,
        )
    for fraction in np.linspace(0.0, 1.0, 6):
        value = fraction * x_max
        x = left + int(round(fraction * plot_width))
        cv2.line(canvas, (x, top), (x, top + plot_height), (230, 230, 230), 1)
        cv2.putText(
            canvas, f"{value:.2f}", (x - 24, top + plot_height + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 50), 1, cv2.LINE_AA,
        )
    for name, values in (
        ("5-fold held-out BC error", cv_error),
        ("LOEO recovery-actor error", loeo_error),
        ("cross-episode NN action disagreement", nn_disagreement),
    ):
        x_values, y_values = _cdf_points(values)
        points = np.asarray(
            [
                (
                    left + int(round(min(value, x_max) / x_max * plot_width)),
                    top + plot_height - int(round(cdf * plot_height)),
                )
                for value, cdf in zip(x_values, y_values, strict=True)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, colors[name], 3, cv2.LINE_AA)
    candidate_label_y = top + 35
    for label, candidate in candidates.items():
        radius = float(candidate["radius"])
        x = left + int(round(min(radius, x_max) / x_max * plot_width))
        cv2.line(
            canvas, (x, top), (x, top + plot_height), (100, 100, 100), 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"{label}: d={radius:.3f}", (x + 5, candidate_label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA,
        )
        candidate_label_y += 23
    cv2.rectangle(
        canvas, (left, 80), (left + plot_width, 80 + plot_height),
        (40, 40, 40), 2,
    )
    cv2.putText(
        canvas, "Limit-action radius calibration: empirical CDF",
        (left, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "normalized action L2 distance", (width // 2 - 155, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "CDF", (17, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (25, 25, 25), 2, cv2.LINE_AA,
    )
    legend_y = height - 82
    legend_x = left
    for name, color in colors.items():
        cv2.line(canvas, (legend_x, legend_y), (legend_x + 35, legend_y), color, 4)
        cv2.putText(
            canvas, name, (legend_x + 43, legend_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA,
        )
        legend_x += 405
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"failed to write plot: {output}")


def write_transition_csv(
    output: Path,
    data: EncodedTeachingData,
    *,
    fold_ids: np.ndarray,
    cv_error: np.ndarray,
    loeo_error: np.ndarray,
    nearest_indices: np.ndarray,
    nearest_state_distance: np.ndarray,
    nearest_action_disagreement: np.ndarray,
) -> None:
    episode = data.episode_ids.numpy()
    step = data.step_ids.numpy()
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "episode_id", "step_id", "cv_fold", "cv_bc_l2",
                "loeo_recovery_l2", "nn_episode_id", "nn_step_id",
                "nn_state_distance", "nn_action_disagreement_l2",
            )
        )
        for index in range(len(cv_error)):
            neighbor = int(nearest_indices[index])
            writer.writerow(
                (
                    int(episode[index]), int(step[index]), int(fold_ids[index]),
                    float(cv_error[index]), float(loeo_error[index]),
                    int(episode[neighbor]), int(step[neighbor]),
                    float(nearest_state_distance[index]),
                    float(nearest_action_disagreement[index]),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="run containing buffer/*.pkl initial demos"
    )
    parser.add_argument("--task", default="hang_double_strings_2")
    parser.add_argument("--episode-count", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--loeo-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--split-seed", type=int, default=20260904)
    parser.add_argument("--model-seed", type=int, default=4109)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.episode_count < args.folds:
        parser.error("episode-count must be at least folds")
    output_dir = args.output_dir or (
        args.run / "analysis" / "limit_action_threshold_calibration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    modeling = load_component(args.task, "modeling")
    task = modeling.Modeling(config=modeling.ModelingConfig(device=args.device))
    episodes = load_initial_teaching_episodes(
        args.run / "buffer", episode_count=args.episode_count
    )
    print(
        f"Encoding {len(episodes)} teaching episodes / "
        f"{sum(map(len, episodes))} transitions...",
        flush=True,
    )
    data = encode_episodes(task, episodes, encode_batch_size=args.encode_batch_size)

    cv_error, fold_ids, fold_reports = run_episode_cross_validation(
        data,
        folds=args.folds,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        mechanism_dim=task.config.mechanism_obs_dims,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
    )
    loeo_error, loeo_reports = run_leave_one_episode_out(
        data,
        model_seed=args.model_seed + 10_000,
        mechanism_dim=task.config.mechanism_obs_dims,
        device=device,
        epochs=args.loeo_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
    )
    nearest, nn_state_distance, nn_action_disagreement = (
        cross_episode_nearest_neighbor_disagreement(
            data, device=device, query_batch_size=args.batch_size
        )
    )
    candidates = candidate_thresholds(cv_error, sigma=ACTION_LIKENESS_SIGMA)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "source": {
            "run": str(args.run),
            "buffer": str(args.run / "buffer"),
            "task": args.task,
        },
        "dataset": {
            "episode_count": len(data.episode_lengths),
            "transition_count": len(data.actions),
            "episode_lengths": list(data.episode_lengths),
            "observation_dim": int(data.observations.shape[1]),
            "action_dim": int(data.actions.shape[1]),
            "requirements": "first complete successful fully-human episodes only",
        },
        "configuration": {
            "folds": args.folds,
            "epochs": args.epochs,
            "loeo_epochs": args.loeo_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "encoder_dim": args.encoder_dim,
            "hidden_dim": args.hidden_dim,
            "split_seed": args.split_seed,
            "model_seed": args.model_seed,
            "action_likeness_sigma": ACTION_LIKENESS_SIGMA,
            "previous_runtime_radius": PREVIOUS_ACTION_LIKENESS_RADIUS,
            "existing_h_score_threshold": radius_to_h_score(
                PREVIOUS_ACTION_LIKENESS_RADIUS, sigma=ACTION_LIKENESS_SIGMA
            ),
        },
        "primary_5_fold_episode_cv": {
            "error": error_summary(cv_error),
            "existing_radius_held_out_coverage": float(
                np.mean(cv_error <= PREVIOUS_ACTION_LIKENESS_RADIUS)
            ),
            "candidate_thresholds": candidates,
            "folds": fold_reports,
        },
        "diagnostics": {
            "cross_episode_nearest_neighbor": {
                "observation_metric": (
                    "RMS Euclidean distance after per-dimension standardization; "
                    "near-constant dimensions omitted"
                ),
                "action_disagreement": error_summary(nn_action_disagreement),
                "nearest_state_distance": error_summary(nn_state_distance),
            },
            "leave_one_episode_out_recovery_actor": {
                "training": (
                    "same deterministic recovery-actor architecture, supervised "
                    "by human-action MSE on the other 19 episodes"
                ),
                "error": error_summary(loeo_error),
                "episodes": loeo_reports,
            },
        },
        "artifacts": {
            "transition_table": "transition_errors.csv",
            "empirical_cdf": "empirical_cdf.png",
        },
    }
    with (output_dir / "calibration.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_transition_csv(
        output_dir / "transition_errors.csv",
        data,
        fold_ids=fold_ids,
        cv_error=cv_error,
        loeo_error=loeo_error,
        nearest_indices=nearest,
        nearest_state_distance=nn_state_distance,
        nearest_action_disagreement=nn_action_disagreement,
    )
    draw_empirical_cdf(
        output_dir / "empirical_cdf.png",
        cv_error=cv_error,
        loeo_error=loeo_error,
        nn_disagreement=nn_action_disagreement,
        candidates=candidates,
    )
    print(json.dumps(report["primary_5_fold_episode_cv"], indent=2), flush=True)
    print(f"Saved calibration artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
