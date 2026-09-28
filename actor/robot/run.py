"""Run the task-owned UR5e driver without importing an algorithm."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tasks import load_component

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hang_double_strings_2", help="package name below tasks/")
    parser.add_argument("--request-host", default="0.0.0.0")
    parser.add_argument("--request-port", type=int, default=7001)
    parser.add_argument("--response-host", default="127.0.0.1")
    parser.add_argument("--response-port", type=int, default=7008)
    parser.add_argument("--raw-images", action="store_true", help="include full global RGB/depth frames in observations")
    parser.add_argument(
        "--global-video",
        type=Path,
        default=None,
        help="override default videos/<timestamp>/global_camera.mp4 segment basename",
    )
    parser.add_argument(
        "--no-global-video",
        action="store_true",
        help="disable the default continuous raw-global-camera recording",
    )
    parser.add_argument(
        "--global-video-anchor-frames",
        type=int,
        default=30,
        help="write one wall-clock timestamp anchor per this many encoded frames",
    )
    parser.add_argument(
        "--control-log",
        type=Path,
        default=None,
        help="optional standalone high-rate controller JSONL log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.global_video_anchor_frames <= 0:
        raise ValueError("--global-video-anchor-frames must be positive")
    global_video_path = None
    if not args.no_global_video:
        global_video_path = (
            args.global_video
            if args.global_video is not None
            else Path("videos")
            / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
            / "global_camera.mp4"
        )
    # Hardware SDK imports are intentionally deferred: `--help` and task
    # discovery must work on learner/actor machines without RealSense.
    from shared.zmq import ZmqEndpointConfig

    component = load_component(args.task, "robot")
    server = component.Robot(
        config=component.RobotConfig(
            request_endpoint=ZmqEndpointConfig(host=args.request_host, port=args.request_port),
            response_endpoint=ZmqEndpointConfig(host=args.response_host, port=args.response_port),
            include_raw_images=args.raw_images,
            global_video_path=global_video_path,
            global_video_anchor_every_frames=args.global_video_anchor_frames,
            control_log_path=args.control_log,
        )
    )
    server.run()


if __name__ == "__main__":
    main()
