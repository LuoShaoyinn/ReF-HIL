#!/usr/bin/env python3
"""Interactively label saved classifier RGB PNGs as success, failure, or skip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="image directory or one classifier RGB PNG")
    parser.add_argument("--output", type=Path, required=True, help="append-only JSONL label file")
    parser.add_argument("--pattern", default="*_classifier_rgb.png", help="recursive image glob")
    parser.add_argument(
        "--condition",
        default=None,
        help=(
            "named condition convenience; uses *_classifier_<name>_rgb.png "
            "(the primary 'success' condition keeps *_classifier_rgb.png)"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    if args.condition is not None:
        args.pattern = (
            "*_classifier_rgb.png"
            if args.condition == "success"
            else f"*_classifier_{args.condition}_rgb.png"
        )

    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob(args.pattern))
    if not paths:
        raise SystemExit(f"no images found under {args.input} with pattern {args.pattern!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            try:
                row = json.loads(line)
                labels[str(row["image"])] = str(row["label"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    index = max(0, min(args.start, len(paths) - 1))
    print(
        "Tagger: y=success, n=failure, s=skip, q=quit; "
        "left/right or a/d navigate."
    )
    try:
        while 0 <= index < len(paths):
            path = paths[index]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"cannot read {path}")
                index += 1
                continue
            display = cv2.resize(
                image,
                (960, max(1, int(image.shape[0] * 960 / image.shape[1]))),
            )
            text = (
                f"{index + 1}/{len(paths)}  {path}  "
                f"label={labels.get(str(path), '-')}"
            )
            cv2.putText(
                display,
                text[:140],
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("SRT classifier image tagger", display)
            # waitKeyEx preserves the extended key codes used by arrow keys.
            # OpenCV reports different values across macOS, X11 and Windows,
            # so retain a/d as portable navigation aliases as well.
            key = cv2.waitKeyEx(0)
            key_byte = key & 0xFF
            if key_byte in (ord("q"), 27):
                break
            if key_byte in (ord("y"), ord("n"), ord("s")):
                label = {
                    ord("y"): "success",
                    ord("n"): "failure",
                    ord("s"): "skip",
                }[key_byte]
                labels[str(path)] = label
                with args.output.open("a") as stream:
                    stream.write(json.dumps({"image": str(path), "label": label}) + "\n")
                index += 1
            elif key_byte == ord("a") or key in {
                81,       # legacy OpenCV
                63234,    # macOS
                65361,    # X11
                2424832,  # Windows
            }:
                index = max(0, index - 1)
            elif key_byte == ord("d") or key in {
                83,       # legacy OpenCV
                63235,    # macOS
                65363,    # X11
                2555904,  # Windows
            }:
                index += 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
