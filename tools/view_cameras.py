#!/usr/bin/env python3
"""Display the configured RealSense RGB streams and depth streams."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WINDOW = "SRT cameras"


def _tile(images: dict[str, np.ndarray], tile_width: int) -> np.ndarray:
    if not images:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    tiles: list[np.ndarray] = []
    for name, image in images.items():
        if image.ndim == 2:
            image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
        image = cv2.resize(image, (tile_width, int(tile_width * image.shape[0] / image.shape[1])))
        cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 26), (0, 0, 0), -1)
        cv2.putText(image, name, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    columns = 2 if len(tiles) <= 4 else math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    tile_height = max(tile.shape[0] for tile in tiles)
    blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    rows_out: list[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        row = [cv2.copyMakeBorder(t, 0, tile_height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in row]
        rows_out.append(cv2.hconcat(row))
    return cv2.vconcat(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", help="camera name; repeat or omit for all configured cameras")
    parser.add_argument("--width", type=int, default=480, help="display tile width in pixels")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    try:
        from actor.robot.ur5e import RealSenseCamera, UR5eBaseConfig
    except ModuleNotFoundError as exc:
        if exc.name == "pyrealsense2":
            raise SystemExit(
                "pyrealsense2 is required for the camera viewer; install/run with the project actor extra"
            ) from exc
        raise

    config = UR5eBaseConfig()
    selected = set(args.camera or config.realsense.keys())
    cameras = {name: RealSenseCamera(camera_config) for name, camera_config in config.realsense.items() if name in selected}
    if not cameras:
        raise SystemExit(f"No configured camera matches {sorted(selected)}")
    try:
        for name, camera in cameras.items():
            camera.connect()
            print(f"connected {name}: depth={camera.use_depth}")
        print("Press q or Esc to close.")
        while True:
            images: dict[str, np.ndarray] = {}
            for name, camera in cameras.items():
                camera.update()
                images[name] = camera.read_image()
                if camera.use_depth:
                    images[f"{name}_depth"] = camera.read_depth()
            cv2.imshow(WINDOW, _tile(images, args.width))
            key = cv2.waitKey(max(1, int(1000.0 / args.fps))) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for camera in cameras.values():
            camera.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
