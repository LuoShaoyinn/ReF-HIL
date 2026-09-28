"""Run the branch-local action-limited SAC actor."""

from __future__ import annotations

import argparse
from pathlib import Path

from actor.transport import (
    FakeOperatorClient,
    FakeRobotClient,
    ZmqOperatorClient,
    ZmqRobotClient,
)
from actor.runtime import Actor, ActorConfig
from actor.success import TaskSuccessOracle
from actor.terminal_success import TerminalSuccessKey
from shared.actor_network import ActorInferencePolicy, ActorNetworkConfig
from shared.zmq import ZmqEndpointConfig
from tasks import load_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hang_double_strings_2", help="package name below tasks/")
    parser.add_argument("--experiment-name", default="unnamed")
    parser.add_argument("--learner-host", default="192.168.1.106")
    parser.add_argument("--learner-port", type=int, default=7003)
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=7001)
    parser.add_argument("--robot-response-host", default="0.0.0.0")
    parser.add_argument("--robot-response-port", type=int, default=7008)
    parser.add_argument("--spacemouse-host", default="127.0.0.1")
    parser.add_argument("--spacemouse-port", type=int, default=7002)
    parser.add_argument("--spacemouse-response-host", default="0.0.0.0")
    parser.add_argument("--spacemouse-response-port", type=int, default=7003)
    parser.add_argument("--parameters-host", default="0.0.0.0")
    parser.add_argument("--parameters-port", type=int, default=7004)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--classifier-device", default="cuda")
    parser.add_argument("--fake-robot", action="store_true")
    parser.add_argument("--fake-spacemouse", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=1000000)
    parser.add_argument(
        "--save-episode-actors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the exact policy used by every physical episode",
    )
    parser.add_argument("--episode-actor-dir", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override tasks/<task>/config.py max_episode_steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modeling = load_component(args.task, "modeling")
    task = modeling.Modeling(config=modeling.ModelingConfig(device=args.policy_device))
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
    classifier = None
    if not args.fake_robot:
        classifier = TaskSuccessOracle.load(
            task_name=args.task,
            task=task.config.task,
            experiment_name=args.experiment_name,
            device=args.classifier_device,
        )
    robot = (
        FakeRobotClient(
            image_size=task.config.task.policy_image_size,
            linear_speed=task.config.task.linear_speed,
            initial_position=task.config.task.reset_pos,
            initial_rotation=task.config.task.zero_point_rot,
            gripper_position_range=task.config.task.gripper_position_range,
        )
        if args.fake_robot
        else ZmqRobotClient(
            request_endpoint=ZmqEndpointConfig(
                host=args.robot_host, port=args.robot_port
            ),
            response_endpoint=ZmqEndpointConfig(
                host=args.robot_response_host, port=args.robot_response_port
            ),
        )
    )
    operator = (
        FakeOperatorClient()
        if args.fake_spacemouse
        else ZmqOperatorClient(
            request_endpoint=ZmqEndpointConfig(
                host=args.spacemouse_host, port=args.spacemouse_port
            ),
            response_endpoint=ZmqEndpointConfig(
                host=args.spacemouse_response_host,
                port=args.spacemouse_response_port,
            ),
        )
    )
    with TerminalSuccessKey(enabled=not args.fake_robot) as success_key:
        actor = Actor(
            config=ActorConfig(
                learner_endpoint=ZmqEndpointConfig(
                    host=args.learner_host, port=args.learner_port
                ),
                parameters_endpoint=ZmqEndpointConfig(
                    host=args.parameters_host, port=args.parameters_port
                ),
                max_episodes=args.max_episodes,
                max_steps_per_episode=(
                    task.config.task.max_episode_steps
                    if args.max_steps is None
                    else args.max_steps
                ),
                default_not_success_reward=-float(task.config.task.step_penalty),
                wait_for_initial_parameters=not args.fake_robot,
                episode_actor_directory=(
                    None
                    if not args.save_episode_actors or args.fake_robot
                    else Path("outputs") / args.experiment_name / "episode_actors"
                    if args.episode_actor_dir is None
                    else args.episode_actor_dir
                ),
            ),
            task=task,
            policy=policy,
            robot=robot,
            operator=operator,
            success_oracle=(
                (lambda _raw_observation: False)
                if classifier is None
                else lambda raw_observation: classifier.inference(
                    raw_robot_observation=raw_observation
                )
            ),
            force_success_requested=success_key.success_requested,
            force_failure_requested=success_key.failure_requested,
            drain_force_success=success_key.drain,
        )
        actor.run()


if __name__ == "__main__":
    main()
