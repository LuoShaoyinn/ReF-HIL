"""Train a validator-compatible deterministic BC actor from all human rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import torch
import torch.nn.functional as F

from shared.actor_network import (
    ACTOR_SNAPSHOT_FORMAT,
    ActorNetworkConfig,
    DeterministicActorHead,
    ObservationEncoder,
)
from tasks import load_component


def encode_all_human_rows(
    *,
    buffer_directory: Path,
    task: object,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    pending: list[dict] = []
    transition_count = 0
    human_count = 0
    episode_count = 0
    human_episode_ids: set[int] = set()
    paths = sorted(buffer_directory.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no replay pickle files in {buffer_directory}")

    def flush() -> None:
        if not pending:
            return
        observations.append(
            task.build_observations(
                [row["raw_obs"] for row in pending],
                [row["info"] for row in pending],
                augment=False,
            ).detach().cpu()
        )
        actions.append(
            task.build_actions(
                [row["raw_action"] for row in pending],
                [row["info"] for row in pending],
                augment=False,
            ).detach().cpu()
        )
        pending.clear()

    for path_index, path in enumerate(paths, 1):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"expected a transition list in {path}")
        for row in payload:
            transition_count += 1
            if bool(row.get("info", {}).get("is_intervene", False)):
                pending.append(row)
                human_count += 1
                human_episode_ids.add(episode_count)
                if len(pending) >= batch_size:
                    flush()
            if bool(row.get("done", False)):
                episode_count += 1
        if path_index == 1 or path_index % 20 == 0 or path_index == len(paths):
            print(
                f"Encoded replay {path_index}/{len(paths)}: "
                f"rows={transition_count} human={human_count} episodes={episode_count}",
                flush=True,
            )
    flush()
    if not observations:
        raise RuntimeError("the replay contains no is_intervene=True rows")
    return torch.cat(observations), torch.cat(actions), {
        "replay_transitions": transition_count,
        "complete_episodes": episode_count,
        "human_transitions": human_count,
        "episodes_with_human_actions": len(human_episode_ids),
    }


@torch.inference_mode()
def evaluate(
    encoder: ObservationEncoder,
    actor: DeterministicActorHead,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, object]:
    predictions: list[torch.Tensor] = []
    for start in range(0, len(observations), batch_size):
        observation = observations[start : start + batch_size].to(device)
        predictions.append(actor(encoder(observation)).cpu())
    prediction = torch.cat(predictions)
    error = prediction - actions
    mae = error.abs().mean(dim=0)
    rmse = error.square().mean(dim=0).sqrt()
    correlations: list[float] = []
    for dimension in range(actions.shape[1]):
        target = actions[:, dimension]
        estimate = prediction[:, dimension]
        if target.std() == 0 or estimate.std() == 0:
            correlations.append(float("nan"))
        else:
            correlations.append(float(torch.corrcoef(torch.stack((target, estimate)))[0, 1]))
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "per_dimension_mae": [float(value) for value in mae],
        "per_dimension_rmse": [float(value) for value in rmse],
        "per_dimension_correlation": correlations,
        "gripper_sign_accuracy": float(
            (prediction[:, -1].sign() == actions[:, -1].sign()).float().mean()
        ),
    }


def actor_snapshot(
    *,
    config: ActorNetworkConfig,
    encoder: ObservationEncoder,
    actor: DeterministicActorHead,
) -> dict:
    return {
        "format": ACTOR_SNAPSHOT_FORMAT,
        "config": config.as_dict(),
        "encoder": {
            name: value.detach().cpu().numpy()
            for name, value in encoder.state_dict().items()
        },
        "actor": {
            name: value.detach().cpu().numpy()
            for name, value in actor.state_dict().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20_260_824)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.encode_batch_size < 1:
        parser.error("updates and batch sizes must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_directory = Path("outputs") / args.source_experiment
    output_directory = (
        run_directory / "bc_all_human"
        if args.output_dir is None
        else args.output_dir
    )
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    modeling = load_component(args.task, "modeling")
    task = modeling.Modeling(config=modeling.ModelingConfig(device=args.device))
    observations, actions, data_metrics = encode_all_human_rows(
        buffer_directory=run_directory / "buffer",
        task=task,
        batch_size=args.encode_batch_size,
    )
    config = ActorNetworkConfig(
        obs_dims=task.config.obs_dims,
        mechanism_obs_dims=task.config.mechanism_obs_dims,
        encoder_dim=256,
        hidden_dim=256,
        action_dims=task.config.action_dims,
    )
    encoder = ObservationEncoder(
        config.obs_dims,
        config.mechanism_obs_dims,
        config.encoder_dim,
    ).to(device)
    actor = DeterministicActorHead(
        encoder.output_dim,
        config.hidden_dim,
        config.action_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *actor.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    observations = observations.pin_memory()
    actions = actions.pin_memory()
    generator = torch.Generator().manual_seed(args.seed + 1)
    history: list[dict[str, float | int]] = []
    encoder.train()
    actor.train()
    for update in range(1, args.updates + 1):
        indices = torch.randint(
            len(observations),
            (args.batch_size,),
            generator=generator,
        )
        observation = observations[indices].to(device, non_blocking=True)
        target = actions[indices].to(device, non_blocking=True)
        prediction = actor(encoder(observation))
        loss = F.smooth_l1_loss(prediction, target, beta=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*encoder.parameters(), *actor.parameters()], 10.0
        )
        optimizer.step()
        if update == 1 or update % 500 == 0 or update == args.updates:
            record = {"update": update, "loss": float(loss.detach())}
            history.append(record)
            print(
                f"BC update {update}/{args.updates}: loss={record['loss']:.6f}",
                flush=True,
            )

    encoder.eval()
    actor.eval()
    fit_metrics = evaluate(
        encoder,
        actor,
        observations,
        actions,
        device=device,
    )
    checkpoint = {
        "format_version": 1,
        "class_name": "OfflineBehaviorCloning",
        "elapsed_seconds": 0,
        "learner": {
            "train_steps": args.updates,
            "actor_steps": data_metrics["human_transitions"],
            "actor_episodes": data_metrics["episodes_with_human_actions"],
        },
        "actor": actor_snapshot(config=config, encoder=encoder, actor=actor),
        "offline_bc": {
            "task": args.task,
            "source_experiment": args.source_experiment,
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "data": data_metrics,
            "fit": fit_metrics,
        },
    }
    checkpoint_path = checkpoint_directory / "checkpoint_00000000s.pkl"
    temporary = checkpoint_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
    temporary.replace(checkpoint_path)
    metrics = {
        **checkpoint["offline_bc"],
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = output_directory / "metrics.json"
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    with temporary_metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_metrics.replace(metrics_path)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
