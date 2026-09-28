#!/usr/bin/env python3
"""Show the full-resolution RGB image from the configured global camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAMERA_WINDOW = "SRT global camera"
CROP_WINDOW = "SRT global camera crop"
Crop = tuple[int, int, int, int]


def normalize_crop(crop: Crop, *, width: int, height: int) -> Crop:
    """Clamp an x0/y0/x1/y1 crop, with x1/y1 as exclusive bounds."""

    x0, y0, x1, y1 = crop
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    x1 = int(np.clip(x1, x0 + 1, width))
    y1 = int(np.clip(y1, y0 + 1, height))
    return x0, y0, x1, y1


def xywh_to_corners(crop: Crop) -> Crop:
    """Convert task-config (x, y, width, height) into exclusive corners."""

    x, y, width, height = (int(value) for value in crop)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid task crop {crop}")
    return x, y, x + width, y + height


def draw_configured_crop(
    image: np.ndarray,
    crop: Crop,
    *,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Overlay one task-config crop on a raw camera image."""

    x0, y0, x1, y1 = normalize_crop(
        xywh_to_corners(crop), width=image.shape[1], height=image.shape[0]
    )
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2)
    cv2.putText(
        image,
        f"{label}: {crop}",
        (x0 + 4, max(18, y0 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


class CropSelector:
    def __init__(self, initial_crop: Crop | None = None) -> None:
        self.crop = initial_crop
        self.anchor: tuple[int, int] | None = None
        self.cursor: tuple[int, int] | None = None
        self.width = 0
        self.height = 0

    def set_image_shape(self, image: np.ndarray) -> None:
        self.height, self.width = image.shape[:2]
        if self.crop is not None:
            self.crop = normalize_crop(self.crop, width=self.width, height=self.height)

    def clear(self) -> None:
        self.crop = None
        self.anchor = None
        self.cursor = None

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.width <= 0 or self.height <= 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.anchor = (x, y)
            self.cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.anchor is not None:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.anchor is not None:
            self.cursor = (x, y)
            ax, ay = self.anchor
            # Mouse endpoints are inclusive; array crop upper bounds are exclusive.
            self.crop = normalize_crop(
                (min(ax, x), min(ay, y), max(ax, x) + 1, max(ay, y) + 1),
                width=self.width,
                height=self.height,
            )
            self.anchor = None
            self.cursor = None
            x0, y0, x1, y1 = self.crop
            print(
                f"crop: x=[{x0},{x1}) y=[{y0},{y1}) size={x1 - x0}x{y1 - y0}\n"
                f"task config tuple: ({x0}, {y0}, {x1 - x0}, {y1 - y0})\n"
                f"reuse with: --crop {x0} {y0} {x1} {y1}",
                flush=True,
            )

    def display_crop(self) -> Crop | None:
        if self.anchor is None or self.cursor is None:
            return self.crop
        ax, ay = self.anchor
        x, y = self.cursor
        return normalize_crop(
            (min(ax, x), min(ay, y), max(ax, x) + 1, max(ay, y) + 1),
            width=self.width,
            height=self.height,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default=None,
        help="overlay classifier and global-policy crops from the task config",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--request-host", default="127.0.0.1")
    parser.add_argument("--request-port", type=int, default=7001)
    parser.add_argument("--response-host", default="0.0.0.0")
    parser.add_argument("--response-port", type=int, default=7008)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="initial crop with exclusive X1/Y1 bounds",
    )
    args = parser.parse_args()

    task_config = None
    if args.task is not None:
        from tasks import load_component

        task_config = load_component(args.task, "config").TaskConfig()
        condition_text = ", ".join(
            f"{condition.name}={condition.clip}"
            for condition in task_config.success_conditions
        )
        print(
            f"{args.task} success_conditions=[{condition_text}] "
            f"global_policy_clip={task_config.global_policy_clip}",
            flush=True,
        )

    from shared.zmq import Receiver, Sender, ZmqEndpointConfig

    request = Sender(ZmqEndpointConfig(host=args.request_host, port=args.request_port))
    response = Receiver(
        ZmqEndpointConfig(host=args.response_host, port=args.response_port),
        timeout_ms=args.timeout_ms,
    )
    selector = CropSelector(None if args.crop is None else tuple(args.crop))
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CAMERA_WINDOW, selector.mouse_callback)
    try:
        print("Drag the left mouse button to crop. Press r to reset, q or Esc to close.")
        while True:
            request.send("read_global_camera")
            image_rgb = response.recv(None)
            if image_rgb is None:
                raise RuntimeError("timed out waiting for the robot driver's global camera frame")
            # RealSense is configured as RGB8; OpenCV displays BGR.
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            selector.set_image_shape(image_bgr)
            display = image_bgr.copy()
            if task_config is not None:
                classifier_colors = ((0, 180, 255), (0, 255, 180), (255, 0, 180))
                for index, condition in enumerate(task_config.success_conditions):
                    draw_configured_crop(
                        display,
                        condition.clip,
                        label=f"classifier:{condition.name}",
                        color=classifier_colors[index % len(classifier_colors)],
                    )
                draw_configured_crop(
                    display,
                    task_config.global_policy_clip,
                    label="global policy",
                    color=(255, 180, 0),
                )
            crop = selector.display_crop()
            if crop is not None:
                x0, y0, x1, y1 = crop
                cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
                cv2.imshow(CROP_WINDOW, image_bgr[y0:y1, x0:x1])
            cv2.imshow(CAMERA_WINDOW, display)
            key = cv2.waitKey(max(1, int(1000.0 / args.fps))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                selector.clear()
                try:
                    cv2.destroyWindow(CROP_WINDOW)
                except cv2.error:
                    pass
    finally:
        request.close()
        response.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
