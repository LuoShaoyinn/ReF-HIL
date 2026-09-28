"""Episode-disjoint behavioral-cloning calibration for action support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from learner.action_proximity import (
    RecoveryActorHead,
    proximity_threshold_from_radius,
)
from learner.observation_encoder import ObservationEncoder


@dataclass(frozen=True)
class EncodedTeachingData:
    observations: torch.Tensor
    actions: torch.Tensor
    episode_ids: torch.Tensor
    step_ids: torch.Tensor
    episode_lengths: tuple[int, ...]


class BehaviorCloningActor(nn.Module):
    """The project's deterministic recovery-actor architecture trained as BC."""

    def __init__(
        self,
        *,
        observation_dim: int,
        mechanism_dim: int,
        action_dim: int,
        encoder_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(
            observation_dim, mechanism_dim, encoder_dim
        )
        self.actor = RecoveryActorHead(
            int(self.encoder.output_dim), hidden_dim, action_dim
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.encoder(observation, detach=False))


def episode_folds(
    episode_count: int, *, folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Return episode-disjoint train/validation IDs for K-fold CV."""

    if folds < 2 or folds > episode_count:
        raise ValueError("folds must be in [2, episode_count]")
    shuffled = np.random.default_rng(seed).permutation(episode_count)
    validation_sets = np.array_split(shuffled, folds)
    all_ids = set(range(episode_count))
    result: list[tuple[list[int], list[int]]] = []
    for validation in validation_sets:
        validation_ids = sorted(int(value) for value in validation)
        training_ids = sorted(all_ids.difference(validation_ids))
        if set(training_ids).intersection(validation_ids):
            raise AssertionError("episode leakage in cross-validation split")
        result.append((training_ids, validation_ids))
    return result


def radius_to_h_score(radius: float, *, sigma: float = 1.0) -> float:
    return proximity_threshold_from_radius(radius, sigma=sigma)


def error_summary(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("errors must be a non-empty finite vector")
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q97_5": float(np.quantile(values, 0.975)),
        "maximum": float(values.max()),
    }


def _indices_for_episodes(
    episode_ids: torch.Tensor, selected: Iterable[int]
) -> torch.Tensor:
    selected_tensor = torch.as_tensor(list(selected), dtype=episode_ids.dtype)
    return torch.isin(episode_ids, selected_tensor).nonzero().reshape(-1)


def train_bc_actor(
    data: EncodedTeachingData,
    training_episode_ids: list[int],
    *,
    mechanism_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    encoder_dim: int,
    hidden_dim: int,
    seed: int,
) -> tuple[BehaviorCloningActor, float]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = BehaviorCloningActor(
        observation_dim=data.observations.shape[1],
        mechanism_dim=mechanism_dim,
        action_dim=data.actions.shape[1],
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_indices = _indices_for_episodes(
        data.episode_ids, training_episode_ids
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 100_003)
    final_loss = float("nan")
    model.train()
    for _epoch in range(epochs):
        permutation = training_indices[
            torch.randperm(len(training_indices), generator=generator)
        ]
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            observation = data.observations[indices].to(device, non_blocking=True)
            target = data.actions[indices].to(device, non_blocking=True)
            prediction = model(observation)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(indices)
            loss_sum += float(loss.detach()) * count
            sample_count += count
        final_loss = loss_sum / sample_count
    model.eval()
    return model, final_loss


@torch.inference_mode()
def prediction_errors(
    model: BehaviorCloningActor,
    data: EncodedTeachingData,
    evaluation_episode_ids: list[int],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = _indices_for_episodes(data.episode_ids, evaluation_episode_ids)
    predictions: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        predictions.append(
            model(data.observations[selected].to(device, non_blocking=True)).cpu()
        )
    prediction = torch.cat(predictions)
    error = torch.linalg.vector_norm(prediction - data.actions[indices], dim=1)
    return indices, error


def run_episode_cross_validation(
    data: EncodedTeachingData,
    *,
    folds: int,
    split_seed: int,
    model_seed: int,
    mechanism_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    encoder_dim: int,
    hidden_dim: int,
    progress_label: str = "5-fold BC",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    errors = np.full(len(data.actions), np.nan, dtype=np.float64)
    fold_ids = np.full(len(data.actions), -1, dtype=np.int64)
    fold_reports: list[dict] = []
    splits = episode_folds(len(data.episode_lengths), folds=folds, seed=split_seed)
    for fold, (training, validation) in enumerate(splits):
        print(
            f"[{progress_label}] fold {fold + 1}/{folds}: "
            f"train={training} validation={validation}",
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
            seed=model_seed + fold,
        )
        indices, fold_error = prediction_errors(
            model, data, validation, device=device, batch_size=batch_size
        )
        fold_values = fold_error.numpy().astype(np.float64)
        errors[indices.numpy()] = fold_values
        fold_ids[indices.numpy()] = fold
        fold_reports.append(
            {
                "fold": fold,
                "training_episode_ids": training,
                "validation_episode_ids": validation,
                "final_training_mse": training_loss,
                "validation_error": error_summary(fold_values),
            }
        )
        del model
    if not np.isfinite(errors).all() or bool((fold_ids < 0).any()):
        raise AssertionError("cross-validation did not evaluate every transition once")
    return errors, fold_ids, fold_reports
