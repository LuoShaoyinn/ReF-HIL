"""Autonomously validate every elapsed-time learner checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from actor.transport import ZmqRobotClient
from actor.episode_snapshots import (
    episode_actor_named,
    episode_actors_from,
    list_episode_actors,
)
from actor.validation import (
    CheckpointValidator,
    TerminalSkipKey,
    ValidationConfig,
    checkpoints_from,
    list_checkpoints,
)
from actor.success import TaskSuccessOracle
from shared.actor_network import ActorInferencePolicy, ActorNetworkConfig
from shared.zmq import ZmqEndpointConfig
from tasks import load_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hang_double_strings_2")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--episode-actor-dir",
        type=Path,
        default=None,
        help="validate exact per-training-episode actor snapshots",
    )
    parser.add_argument(
        "--start-checkpoint",
        default=None,
        help=(
            "start inclusively at these elapsed seconds or this "
            "checkpoint_########s.pkl filename"
        ),
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--start-actor-episode", type=int, default=None)
    parser.add_argument(
        "--actor-name",
        default=None,
        help=(
            "validate exactly one saved actor, e.g. "
            "actor_episode_000062.pkl or 62"
        ),
    )
    parser.add_argument("--episodes-per-checkpoint", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=7001)
    parser.add_argument("--robot-response-host", default="0.0.0.0")
    parser.add_argument("--robot-response-port", type=int, default=7008)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--classifier-device", default="cuda")
    parser.add_argument("--skip-key", default="n")
    parser.add_argument(
        "--interactive-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow one keypress to mark the current attempt failed and reset",
    )
    return parser.parse_args()


def next_validation_directory(base: Path, artifact: Path) -> Path:
    """Allocate a new monotonically numbered validation directory."""

    base.mkdir(parents=True, exist_ok=True)
    prefix = f"{artifact.stem}-"
    used_ids: set[int] = set()
    for candidate in base.glob(f"{prefix}*"):
        suffix = candidate.name.removeprefix(prefix)
        if candidate.is_dir() and suffix.isdigit():
            used_ids.add(int(suffix))
    validation_id = max(used_ids, default=-1) + 1
    result = base / f"{prefix}{validation_id:03d}"
    result.mkdir(parents=False, exist_ok=False)
    return result


def print_checkpoint_result(summary: dict, result_dir: Path) -> None:
    checkpoint = summary["checkpoints"][-1]
    print(
        "[VALIDATE] finished "
        f"checkpoint={checkpoint['checkpoint']} "
        f"success={checkpoint['successes']}/{checkpoint['episodes']} "
        f"success_rate={100.0 * checkpoint['success_rate']:.1f}% "
        f"mean_steps={checkpoint['mean_steps']:.1f} "
        f"mean_return={checkpoint['mean_return']:.3f}\n"
        f"[VALIDATE] saved to {result_dir}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path("outputs") / args.experiment_name
    actor_mode = args.episode_actor_dir is not None or args.actor_name is not None
    if actor_mode and args.checkpoint_dir is not None:
        raise ValueError("use either --episode-actor-dir or --checkpoint-dir")
    if actor_mode and args.start_checkpoint is not None:
        raise ValueError("use --start-actor-episode with --episode-actor-dir")
    if not actor_mode and args.start_actor_episode is not None:
        raise ValueError("--start-actor-episode requires --episode-actor-dir")
    if args.actor_name is not None and args.start_actor_episode is not None:
        raise ValueError("use either --actor-name or --start-actor-episode")
    episode_actor_dir = (
        output_dir / "episode_actors"
        if args.episode_actor_dir is None
        else args.episode_actor_dir
    )
    checkpoint_dir = (
        output_dir / "checkpoints"
        if args.checkpoint_dir is None
        else args.checkpoint_dir
    )
    result_base = (
        Path("outputs") / f"validate-{args.experiment_name}"
        if args.result_dir is None
        else args.result_dir
    )
    episodes_per_checkpoint = (
        20 if args.episodes_per_checkpoint is None else args.episodes_per_checkpoint
    )
    modeling = load_component(args.task, "modeling")
    task = modeling.Modeling(
        config=modeling.ModelingConfig(device=args.policy_device)
    )
    classifier = TaskSuccessOracle.load(
        task_name=args.task,
        task=task.config.task,
        experiment_name=args.experiment_name,
        device=args.classifier_device,
    )
    policy = ActorInferencePolicy(
        device=args.policy_device,
        config=ActorNetworkConfig(
            obs_dims=task.config.obs_dims,
            action_dims=task.config.action_dims,
            mechanism_obs_dims=task.config.mechanism_obs_dims,
            encoder_dim=256,
            hidden_dim=256,
        ),
    )
    with TerminalSkipKey(
        args.skip_key,
        enabled=args.interactive_skip,
    ) as skip_key:
        if not actor_mode:
            artifacts = checkpoints_from(
                list_checkpoints(checkpoint_dir), args.start_checkpoint
            )
        elif args.actor_name is not None:
            artifacts = [episode_actor_named(episode_actor_dir, args.actor_name)]
        else:
            artifacts = episode_actors_from(
                list_episode_actors(episode_actor_dir),
                args.start_actor_episode,
            )
        if not artifacts:
            raise FileNotFoundError("no validation artifacts found")
        for artifact in artifacts:
            result_dir = next_validation_directory(result_base, artifact)
            print(f"[VALIDATE] new run: {result_dir}", flush=True)
            validator = CheckpointValidator(
                config=ValidationConfig(
                    episodes_per_checkpoint=episodes_per_checkpoint,
                    max_steps_per_episode=(
                        task.config.task.max_episode_steps
                        if args.max_steps is None
                        else args.max_steps
                    ),
                    default_not_success_reward=-float(task.config.task.step_penalty),
                ),
                task=task,
                policy=policy,
                robot=ZmqRobotClient(
                    request_endpoint=ZmqEndpointConfig(
                        host=args.robot_host,
                        port=args.robot_port,
                    ),
                    response_endpoint=ZmqEndpointConfig(
                        host=args.robot_response_host,
                        port=args.robot_response_port,
                    ),
                ),
                success_oracle=lambda raw_observation: classifier.inference(
                    raw_robot_observation=raw_observation
                ),
                result_directory=result_dir,
                skip_requested=skip_key.requested,
            )
            summary = validator.run([artifact])
            print_checkpoint_result(summary, result_dir)


if __name__ == "__main__":
    main()
