"""Record rotate-knob classifier images with the task's real reset procedure."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from actor.transport import ZmqOperatorClient, ZmqRobotClient
from shared.zmq import ZmqEndpointConfig
from tools.collect_images_teleop import _next_episode_index, _preview, _save_images

from .modeling import Modeling, ModelingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/classifier_samples/rotate_knob"),
        help="directory containing episode image folders",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="frames per episode; defaults to rotate_knob max_episode_steps",
    )
    parser.add_argument("--period", type=float, default=0.1)
    parser.add_argument(
        "--append",
        action="store_true",
        help="append episode folders without replacing existing samples",
    )
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=7001)
    parser.add_argument("--robot-response-host", default="127.0.0.1")
    parser.add_argument("--robot-response-port", type=int, default=7008)
    parser.add_argument("--spacemouse-host", default="127.0.0.1")
    parser.add_argument("--spacemouse-port", type=int, default=7002)
    parser.add_argument("--spacemouse-response-host", default="127.0.0.1")
    parser.add_argument("--spacemouse-response-port", type=int, default=7003)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")
    if args.period <= 0:
        raise ValueError("period must be positive")

    task = Modeling(config=ModelingConfig())
    task_config = task.config.task
    steps = task_config.max_episode_steps if args.steps is None else args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    first_episode = _next_episode_index(args.output, append=args.append)

    robot = ZmqRobotClient(
        request_endpoint=ZmqEndpointConfig(host=args.robot_host, port=args.robot_port),
        response_endpoint=ZmqEndpointConfig(
            host=args.robot_response_host, port=args.robot_response_port
        ),
    )
    operator = ZmqOperatorClient(
        request_endpoint=ZmqEndpointConfig(
            host=args.spacemouse_host, port=args.spacemouse_port
        ),
        response_endpoint=ZmqEndpointConfig(
            host=args.spacemouse_response_host,
            port=args.spacemouse_response_port,
        ),
    )

    # Modeling.reset intentionally skips the manual countdown at application
    # startup. Classifier collection is episode-oriented, so count the first
    # reset as an inter-episode reset and retain the configured seven seconds.
    task._completed_resets = 1
    try:
        print(
            f"Recording {args.episodes} rotate_knob classifier episodes into "
            f"{args.output}; {steps} frames per episode.",
            flush=True,
        )
        for offset in range(args.episodes):
            episode = first_episode + offset
            directory = args.output / f"episode_{episode:04d}"
            print(
                f"Resetting for classifier episode {offset + 1}/{args.episodes} "
                f"({directory.name}).",
                flush=True,
            )
            task.reset(robot.send_action, robot.read_observation)
            operator.reset()
            last_gripper_command = float(task_config.reset_gripper)
            directory.mkdir(exist_ok=False)

            print(
                "Ready: teleoperate with the SpaceMouse; n ends this episode, "
                "q/Esc exits.",
                flush=True,
            )
            for step in range(steps):
                state = robot.read_observation()
                images = dict(state.get("images", {}))
                required_images = {
                    condition.image_key
                    for condition in task_config.success_conditions
                } | {
                    condition.depth_key
                    for condition in task_config.success_conditions
                }
                missing_images = required_images.difference(images)
                if missing_images:
                    raise RuntimeError(
                        "robot driver is not exposing the rotate_knob classifier "
                        f"crops; missing {sorted(missing_images)}. Start "
                        "actor.robot.run with --task rotate_knob."
                    )
                _save_images(directory, step, images)
                cv2.imshow("SRT rotate_knob classifier demo", _preview(images))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("n"):
                    break

                human_action = operator.read_action(
                    task.build_operator_request(state)
                )
                if not isinstance(human_action, dict):
                    human_action = {
                        "delta_pos": np.zeros(3, dtype=np.float32),
                        "delta_rot": np.zeros(3, dtype=np.float32),
                        "gripper_pressed": False,
                    }
                resolved_action = task.prepare_spacemouse_action(
                    human_action, last_gripper_command
                )
                policy_action = task.build_action_from_spacemouse(
                    resolved_action, {}
                )
                robot_action = task.parse_action(policy_action)
                robot.send_action(robot_action)
                last_gripper_command = float(robot_action["gripper"])
                time.sleep(args.period)
    except KeyboardInterrupt:
        print("Classifier recording interrupted by user.", flush=True)
    finally:
        robot.close()
        operator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
