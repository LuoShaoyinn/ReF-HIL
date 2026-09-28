#!/usr/bin/env python3
"""Drive with the SpaceMouse and save only camera images as PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actor.transport import ZmqOperatorClient, ZmqRobotClient
from shared.task.base import BaseModeling
from shared.task.gripper_calibration import gripper_control_position
from shared.zmq import ZmqEndpointConfig
from tasks import load_component


def _to_opencv_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        # Robot camera arrays are RGB; OpenCV display/write APIs expect BGR.
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _save_images(directory: Path, step: int, images: dict[str, np.ndarray]) -> None:
    for name, image in images.items():
        suffix = "depth" if image.ndim == 2 else "rgb"
        path = directory / f"step_{step:05d}_{name}_{suffix}.png"
        ok, encoded = cv2.imencode(".png", _to_opencv_image(image))
        if not ok:
            raise RuntimeError(f"failed to write {path}")
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())


def _next_episode_index(directory: Path, *, append: bool) -> int:
    indices = []
    for path in directory.glob("episode_*"):
        suffix = path.name.removeprefix("episode_")
        if not path.is_dir() or not suffix.isdecimal():
            raise ValueError(f"unexpected episode path: {path}")
        indices.append(int(suffix))
    if indices and not append:
        raise FileExistsError(
            f"{directory} already contains episodes; use --append to add samples"
        )
    return max(indices, default=-1) + 1


def _preview(images: dict[str, np.ndarray], width: int = 640) -> np.ndarray:
    frames: list[np.ndarray] = []
    for name, image in images.items():
        if image.ndim == 2:
            image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
        else:
            image = _to_opencv_image(image)
        scale = width / image.shape[1]
        frame = cv2.resize(image, (width, max(1, int(image.shape[0] * scale))))
        cv2.putText(frame, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
    return cv2.vconcat(frames) if frames else np.zeros((240, width, 3), dtype=np.uint8)


def _build_robot_action(action: dict, *, allow_rotation: bool) -> dict:
    delta_pos = np.clip(
        np.asarray(action.get("delta_pos", np.zeros(3)), dtype=np.float32), -1.0, 1.0
    )
    delta_rot = np.clip(
        np.asarray(action.get("delta_rot", np.zeros(3)), dtype=np.float32), -1.0, 1.0
    )
    if not allow_rotation:
        delta_rot = np.zeros(3, dtype=np.float32)
    return {
        "delta_pos": delta_pos,
        "delta_rot": delta_rot,
        "gripper": float(np.clip(action.get("gripper", -1.0), -1.0, 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="assemble", help="package name below tasks/")
    parser.add_argument("--output", type=Path, required=True, help="directory containing episode image folders")
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=7001)
    parser.add_argument("--robot-response-host", default="127.0.0.1")
    parser.add_argument("--robot-response-port", type=int, default=7008)
    parser.add_argument("--spacemouse-host", default="127.0.0.1")
    parser.add_argument("--spacemouse-port", type=int, default=7002)
    parser.add_argument("--spacemouse-response-host", default="127.0.0.1")
    parser.add_argument("--spacemouse-response-port", type=int, default=7003)
    parser.add_argument("--append", action="store_true", help="add new episode folders without replacing existing samples")
    parser.add_argument("--episodes", type=int, default=20, help="number of new capture episodes")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--period", type=float, default=0.1)
    args = parser.parse_args()
    task_config = load_component(args.task, "config").TaskConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    first_episode = _next_episode_index(args.output, append=args.append)

    robot = ZmqRobotClient(
        request_endpoint=ZmqEndpointConfig(host=args.robot_host, port=args.robot_port),
        response_endpoint=ZmqEndpointConfig(host=args.robot_response_host, port=args.robot_response_port),
    )
    operator = ZmqOperatorClient(
        request_endpoint=ZmqEndpointConfig(host=args.spacemouse_host, port=args.spacemouse_port),
        response_endpoint=ZmqEndpointConfig(host=args.spacemouse_response_host, port=args.spacemouse_response_port),
    )
    try:
        print("Image-only teleop: q/Esc quits, n skips to the next episode.")
        last_gripper_command = float(task_config.reset_gripper)
        for offset in range(args.episodes):
            episode = first_episode + offset
            directory = args.output / f"episode_{episode:04d}"
            directory.mkdir(exist_ok=False)
            print(f"new episode {offset + 1}/{args.episodes} ({directory.name}): manually reset the robot, then teleoperate")
            for step in range(args.steps):
                state = robot.read_observation()
                images = dict(state.get("images", {})) if isinstance(state, dict) else {}
                _save_images(directory, step, images)
                cv2.imshow("SRT image teleop", _preview(images))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("n"):
                    break

                measured_gripper = gripper_control_position(
                    state, task_config.gripper_position_range
                )
                action = operator.read_action(
                    {"gripper_position": measured_gripper}
                )
                if not isinstance(action, dict):
                    action = {
                        "delta_pos": np.zeros(3, dtype=np.float32),
                        "gripper_pressed": False,
                    }
                resolved_action = BaseModeling.prepare_spacemouse_action(
                    action, last_gripper_command
                )
                robot_action = _build_robot_action(
                    resolved_action,
                    allow_rotation=bool(task_config.allow_rotation),
                )
                robot.send_action(robot_action)
                last_gripper_command = float(robot_action["gripper"])
                time.sleep(args.period)
    finally:
        robot.close()
        operator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
