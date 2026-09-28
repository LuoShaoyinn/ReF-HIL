"""Run the branch-local action-limited SAC learner."""

from __future__ import annotations

import argparse

from learner.action_proximity import ACTION_LIKENESS_SIGMA
from learner.policy import LimitActionPolicy, LimitActionPolicyConfig
from learner.training import LimitActionLearner
from shared.zmq import ZmqEndpointConfig
from tasks import load_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hang_double_strings_2", help="package name below tasks/")
    parser.add_argument("--experiment-name", default="unnamed")
    parser.add_argument("--transitions-host", default="0.0.0.0")
    parser.add_argument("--transitions-port", type=int, default=7003)
    parser.add_argument("--actor-host", default="192.168.1.111")
    parser.add_argument("--actor-parameters-port", type=int, default=7004)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--transitions-before-start", type=int, default=200)
    parser.add_argument("--replay-device", default="cuda")
    parser.add_argument("--max-buffer-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--optimize-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-actor-adam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-seconds", type=float, default=120.0)
    parser.add_argument(
        "--checkpoint-async",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--action-likeness-updates", type=int, default=20_000)
    parser.add_argument("--action-likeness-queries-per-anchor", type=int, default=8)
    parser.add_argument(
        "--action-likeness-sigma",
        type=float,
        choices=(ACTION_LIKENESS_SIGMA,),
        default=ACTION_LIKENESS_SIGMA,
        help="fixed normalized Gaussian-density target width",
    )
    parser.add_argument("--human-iql-expectile", type=float, default=0.75)
    parser.add_argument("--replay-reference-floor-weight", type=float, default=0.10)
    parser.add_argument("--correction-rank-weight", type=float, default=2.0)
    parser.add_argument("--correction-batch-size", type=int, default=256)
    parser.add_argument("--correction-noise-samples", type=int, default=8)
    parser.add_argument("--correction-noise-std", type=float, default=0.15)
    parser.add_argument(
        "--correction-reference-exclusion-radius", type=float, default=0.05
    )
    parser.add_argument("--action-likeness-online-update-every", type=int, default=50)
    parser.add_argument("--recovery-pretrain-updates", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modeling = load_component(args.task, "modeling")
    learner_component = load_component(args.task, "learner")
    task = modeling.Modeling(config=modeling.ModelingConfig(device=args.policy_device))
    policy = LimitActionPolicy(
        config=LimitActionPolicyConfig(
            device=args.policy_device,
            actor_lr=args.actor_lr,
            optimize_actor=args.optimize_actor,
            compile_actor=args.compile_actor,
            fused_actor_adam=args.fused_actor_adam,
            obs_dims=task.config.obs_dims,
            action_dims=task.config.action_dims,
            mechanism_obs_dims=task.config.mechanism_obs_dims,
            horizons=task.config.task.max_episode_steps,
            stay_step_penalty=task.config.task.step_penalty,
            human_iql_expectile=args.human_iql_expectile,
            action_likeness_sigma=args.action_likeness_sigma,
            replay_reference_floor_weight=args.replay_reference_floor_weight,
            correction_rank_weight=args.correction_rank_weight,
            correction_noise_samples=args.correction_noise_samples,
            correction_noise_std=args.correction_noise_std,
            correction_reference_exclusion_radius=(
                args.correction_reference_exclusion_radius
            ),
        )
    )
    learner = LimitActionLearner(
        config=learner_component.LearnerConfig(
            experiment_name=args.experiment_name,
            transitions_endpoint=ZmqEndpointConfig(host=args.transitions_host, port=args.transitions_port),
            actor_parameters_endpoint=ZmqEndpointConfig(
                host=args.actor_host,
                port=args.actor_parameters_port,
            ),
            transitions_before_start=args.transitions_before_start,
            replay_device=args.replay_device,
            max_buffer_size=args.max_buffer_size,
            batch_size=args.batch_size,
            save_every_seconds=args.save_every_seconds,
            checkpoint_async=args.checkpoint_async,
            action_likeness_updates=args.action_likeness_updates,
            action_likeness_queries_per_anchor=(
                args.action_likeness_queries_per_anchor
            ),
            action_likeness_sigma=args.action_likeness_sigma,
            action_likeness_online_update_every=(
                args.action_likeness_online_update_every
            ),
            recovery_pretrain_updates=args.recovery_pretrain_updates,
            correction_batch_size=args.correction_batch_size,
        ),
        task=task,
        policy=policy,
    )
    print(
        f"[ACTOR] lr={policy.actor_optim.param_groups[0]['lr']:g} "
        f"batch={learner.config.batch_size} "
        f"cached={policy.config.optimize_actor} "
        f"fused_adam={policy.actor_optim.param_groups[0]['fused']}",
        flush=True,
    )
    learner.run()


if __name__ == "__main__":
    main()
