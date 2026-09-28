"""Record successful human demonstrations for the branch's task."""

from __future__ import annotations

import argparse
from pathlib import Path

from actor.recorder import Recorder, RecorderConfig, validate_demo_destination
from actor.success import TaskSuccessOracle
from actor.transport import ZmqOperatorClient, ZmqRobotClient
from shared.zmq import ZmqEndpointConfig
from tasks import load_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hang_double_strings_2", help="package name below tasks/")
    parser.add_argument("--experiment-name", default="unnamed")
    parser.add_argument('--append', action='store_true',
                        help='append NEW successful demos to an existing buffer; never overwrite')
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=7001)
    parser.add_argument("--robot-response-host", default="0.0.0.0")
    parser.add_argument("--robot-response-port", type=int, default=7008)
    parser.add_argument("--spacemouse-host", default="127.0.0.1")
    parser.add_argument("--spacemouse-port", type=int, default=7002)
    parser.add_argument("--spacemouse-response-host", default="0.0.0.0")
    parser.add_argument("--spacemouse-response-port", type=int, default=7003)
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="number of NEW successful demonstrations to save; timed-out attempts do not count",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "override the maximum steps in each counted attempt; an attempt "
            "that reaches this limit without success is discarded"
        ),
    )
    parser.add_argument("--classifier-device", default="cuda")
    parser.add_argument(
        "--classifier-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live classifier RGB-D crop, probability, and decision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path('outputs') / args.experiment_name / 'buffer'
    validate_demo_destination(dataset_dir, append=args.append)
    modeling = load_component(args.task, "modeling")
    task = modeling.Modeling(config=modeling.ModelingConfig())
    classifier = TaskSuccessOracle.load(
        task_name=args.task,
        task=task.config.task,
        experiment_name=args.experiment_name,
        device=args.classifier_device,
    )
    recorder = Recorder(
        config=RecorderConfig(
            dataset_dir=dataset_dir,
            append=args.append,
            max_episodes=args.episodes,
            max_steps_per_episode=(
                task.config.task.max_episode_steps
                if args.max_steps is None
                else args.max_steps
            ),
            not_success_reward=-float(task.config.task.step_penalty),
            show_classifier_preview=bool(args.classifier_preview),
        ),
        task=task,
        robot=ZmqRobotClient(
            request_endpoint=ZmqEndpointConfig(host=args.robot_host, port=args.robot_port),
            response_endpoint=ZmqEndpointConfig(host=args.robot_response_host, port=args.robot_response_port),
        ),
        operator=ZmqOperatorClient(
            request_endpoint=ZmqEndpointConfig(host=args.spacemouse_host, port=args.spacemouse_port),
            response_endpoint=ZmqEndpointConfig(
                host=args.spacemouse_response_host,
                port=args.spacemouse_response_port,
            ),
        ),
        success_oracle=lambda raw_observation: classifier.inference(raw_robot_observation=raw_observation),
        success_evaluator=classifier.evaluate,
    )
    recorder.run()


if __name__ == "__main__":
    main()
