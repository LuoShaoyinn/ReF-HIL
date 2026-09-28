#!/usr/bin/env python3
"""Show one or both full-resolution wrist-camera RGB streams."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.zmq import Receiver, Sender, ZmqEndpointConfig  # noqa: E402


CAMERAS = ("wrist_0", "wrist_1")
WINDOW = "SRT wrist camera"


def _read_camera(request: Sender, response: Receiver, name: str) -> np.ndarray:
    request.send(f"read_{name}_camera")
    image = response.recv(None)
    if image is None:
        raise RuntimeError(f"timed out waiting for {name} from the robot driver")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} returned an invalid RGB frame shape: {image.shape}")
    return image


def _draw_crop(
    image_bgr: np.ndarray,
    crop: tuple[int, int, int, int],
    *,
    label: str,
) -> None:
    x, y, width, height = map(int, crop)
    frame_height, frame_width = image_bgr.shape[:2]
    x0 = int(np.clip(x, 0, max(0, frame_width - 1)))
    y0 = int(np.clip(y, 0, max(0, frame_height - 1)))
    x1 = int(np.clip(x + width, x0 + 1, frame_width))
    y1 = int(np.clip(y + height, y0 + 1, frame_height))
    cv2.rectangle(image_bgr, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    cv2.putText(
        image_bgr,
        f"{label} policy crop: {crop}",
        (x0 + 4, max(20, y0 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def _compose(frames: list[tuple[str, np.ndarray]]) -> np.ndarray:
    target_height = min(frame.shape[0] for _, frame in frames)
    panels: list[np.ndarray] = []
    for name, frame in frames:
        if frame.shape[0] != target_height:
            width = max(1, round(frame.shape[1] * target_height / frame.shape[0]))
            frame = cv2.resize(frame, (width, target_height))
        cv2.putText(
            frame,
            f"{name}  {frame.shape[1]}x{frame.shape[0]}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
    return cv2.hconcat(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        choices=(*CAMERAS, "both"),
        default="both",
        help="wrist stream to display",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="optional task name whose wrist policy crops are overlaid",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--request-host", default="127.0.0.1")
    parser.add_argument("--request-port", type=int, default=7001)
    parser.add_argument("--response-host", default="0.0.0.0")
    parser.add_argument("--response-port", type=int, default=7008)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    task = None
    if args.task is not None:
        from tasks import load_component

        task = load_component(args.task, "config").TaskConfig()

    selected = CAMERAS if args.camera == "both" else (args.camera,)
    request = Sender(ZmqEndpointConfig(host=args.request_host, port=args.request_port))
    response = Receiver(
        ZmqEndpointConfig(host=args.response_host, port=args.response_port),
        timeout_ms=args.timeout_ms,
    )
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Showing raw wrist RGB. Press q or Esc to close.", flush=True)
    try:
        while True:
            frames: list[tuple[str, np.ndarray]] = []
            for name in selected:
                rgb = _read_camera(request, response, name)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if task is not None:
                    crop = (
                        task.wrist_0_policy_clip
                        if name == "wrist_0"
                        else task.wrist_1_policy_clip
                    )
                    _draw_crop(bgr, crop, label=name)
                frames.append((name, bgr))
            cv2.imshow(WINDOW, _compose(frames))
            key = cv2.waitKey(max(1, round(1000.0 / args.fps))) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        request.close()
        response.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
