"""Protocol-only actor that periodically pushes complete synthetic episodes.

This is a learner-throughput probe. It emits the same normalized observation
and action payload as the real actor while leaving images as raw uint8 arrays.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from shared.protocol import validate_episode
from shared.zmq import Sender, ZmqEndpointConfig
from tasks import load_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hang_double_strings_2", help="package name below tasks/")
    parser.add_argument("--learner-host", default="127.0.0.1")
    parser.add_argument("--learner-port", type=int, default=7003)
    parser.add_argument("--period-s", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--steps-per-episode", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def _stored_observation(
    *, episode_idx: int, step_idx: int, image_size: int, image_keys: tuple[str, ...]
) -> dict:
    pixel = np.uint8((episode_idx + step_idx) % 256)
    image = np.full((image_size, image_size, 3), pixel, dtype=np.uint8)
    return {
        "tcp_speed": np.zeros((6,), dtype=np.float32),
        "tcp_force": np.zeros((6,), dtype=np.float32),
        "gripper": np.asarray(-1.0, dtype=np.float32),
        "projected_gravity": np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        "images": {key: image for key in image_keys},
    }


def build_episode(
    *,
    episode_idx: int,
    steps: int,
    image_size: int,
    image_keys: tuple[str, ...],
    step_penalty: float = 0.01,
) -> list[dict]:
    if steps < 2:
        raise ValueError("steps-per-episode must be at least 2")
    episode: list[dict] = []
    for step_idx in range(steps):
        done = step_idx == steps - 1
        episode.append(
            {
                "raw_obs": _stored_observation(
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    image_size=image_size,
                    image_keys=image_keys,
                ),
                "raw_action": {
                    "delta_pos": np.zeros((3,), dtype=np.float32),
                    "delta_rot": np.zeros((3,), dtype=np.float32),
                    "gripper": -1.0,
                },
                "reward": 1.0 if done else -float(step_penalty),
                "done": done,
                "info": {
                    "is_intervene": False,
                    "steps_in_episode": step_idx + 1,
                    "max_steps_per_episode": steps,
                    "synthetic": True,
                },
            }
        )
    validate_episode(episode)
    return episode


def main() -> None:
    args = parse_args()
    task = load_component(args.task, "config").TaskConfig()
    steps = task.max_episode_steps if args.steps_per_episode is None else args.steps_per_episode
    image_size = task.policy_image_size if args.image_size is None else args.image_size
    sender = Sender(ZmqEndpointConfig(host=args.learner_host, port=args.learner_port))
    try:
        episode_idx = 0
        while args.episodes <= 0 or episode_idx < args.episodes:
            started = time.monotonic()
            episode = build_episode(
                episode_idx=episode_idx,
                steps=int(steps),
                image_size=int(image_size),
                image_keys=task.image_keys,
                step_penalty=float(task.step_penalty),
            )
            sender.send(episode)
            print(f"[FAKE_ACTOR] sent episode={episode_idx} transitions={len(episode)}")
            episode_idx += 1
            time.sleep(max(0.0, float(args.period_s) - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("[FAKE_ACTOR] interrupted")
    finally:
        sender.close()


if __name__ == "__main__":
    main()
