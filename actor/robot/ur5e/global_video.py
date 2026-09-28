"""Append-only, hardware-encoded global-camera recording."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import numpy as np


class GlobalVideoRecorder:
    """Continuously encode global RGB frames without blocking robot control.

    Each recording is an append-only sequence of independently playable,
    fragmented MP4 segments.  Reusing a destination starts at the next segment
    index rather than overwriting an earlier physical run.  FFmpeg consumes a
    bounded queue in a background thread; encoder/disk stalls can only drop
    video frames, never delay camera acquisition or robot control.
    """

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        queue_size: int = 16,
        anchor_every_frames: int = 30,
        segment_seconds: int = 300,
    ) -> None:
        if min(width, height, fps, queue_size, anchor_every_frames, segment_seconds) <= 0:
            raise ValueError("video dimensions and recorder intervals must be positive")
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.anchor_every_frames = int(anchor_every_frames)
        self.segment_seconds = int(segment_seconds)
        self.anchor_path = self.path.with_suffix(".timestamps.jsonl")
        self.segment_pattern = self.path.with_name(f"{self.path.stem}_%05d.mp4")
        self._frames: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(
            maxsize=int(queue_size)
        )
        self._closed = False
        self.dropped_frames = 0
        self.written_frames = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._writer_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._anchor_log: object | None = None
        self._encoder_errors: deque[str] = deque(maxlen=40)
        self._encoder_failure_reported = False
        self._segment_start_number = 0

    @staticmethod
    def _hardware_encoder_args() -> tuple[str, list[str]]:
        """Return a supported hardware encoder path, preferring VAAPI.

        The robot host exposes h264_vaapi on its AMD GPU.  Other workstations
        may expose AMF instead, so retain it as a fallback.
        """

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required for global-camera recording")
        listed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        ).stdout
        vaapi_device = Path("/dev/dri/renderD128")
        if "h264_vaapi" in listed and vaapi_device.exists():
            return "h264_vaapi", [
                "-vaapi_device",
                str(vaapi_device),
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-b:v",
                "8M",
                "-maxrate",
                "12M",
                "-bufsize",
                "16M",
            ]
        if "h264_amf" in listed:
            return "h264_amf", [
                "-vf",
                "format=rgb0",
                "-c:v",
                "h264_amf",
                "-usage",
                "lowlatency",
                "-quality",
                "speed",
                "-rc",
                "cqp",
                "-qp_i",
                "24",
                "-qp_p",
                "24",
            ]
        raise RuntimeError(
            "no supported hardware H.264 encoder: expected h264_vaapi or h264_amf"
        )

    def _next_segment_number(self) -> int:
        prefix = f"{self.path.stem}_"
        numbers: list[int] = []
        for candidate in self.path.parent.glob(f"{self.path.stem}_*.mp4"):
            suffix = candidate.stem.removeprefix(prefix)
            if suffix.isdecimal():
                numbers.append(int(suffix))
        return max(numbers, default=-1) + 1

    def start(self) -> None:
        if self._process is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._segment_start_number = self._next_segment_number()
        started_at_ns = time.time_ns()
        started_at = datetime.fromtimestamp(
            started_at_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
        log_mode = "a" if self.anchor_path.exists() else "w"
        self._anchor_log = self.anchor_path.open(log_mode, encoding="utf-8", buffering=1)
        self._anchor_log.write(json.dumps({
            "schema": 2,
            "kind": "recording",
            "segment_pattern": self.segment_pattern.name,
            "segment_start_number": self._segment_start_number,
            "start_wall_time_ns": started_at_ns,
            "fps": self.fps,
            "segment_seconds": self.segment_seconds,
            "anchor_every_frames": self.anchor_every_frames,
        }, separators=(",", ":")) + "\n")
        encoder_name, encoder_args = self._hardware_encoder_args()
        ffmpeg = shutil.which("ffmpeg")
        assert ffmpeg is not None
        segment_options = "movflags=+frag_keyframe+empty_moov+default_base_moof"
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                *(encoder_args[:2] if encoder_name == "h264_vaapi" else []),
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                *(encoder_args[2:] if encoder_name == "h264_vaapi" else encoder_args),
                "-force_key_frames",
                f"expr:gte(t,n_forced*{self.segment_seconds})",
                "-f",
                "segment",
                "-segment_time",
                str(self.segment_seconds),
                "-segment_start_number",
                str(self._segment_start_number),
                "-segment_format",
                "mp4",
                "-segment_format_options",
                segment_options,
                "-reset_timestamps",
                "0",
                "-metadata",
                f"creation_time={started_at}",
                "-metadata",
                f"comment=Sparse wall-clock anchors: {self.anchor_path.name}; every {self.anchor_every_frames} encoded frames",
                "-y",
                str(self.segment_pattern),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="global-video-ffmpeg-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        self._writer_thread = threading.Thread(
            target=self._run,
            name="global-video-hardware-writer",
            daemon=True,
        )
        self._writer_thread.start()
        print(
            f"Recording global camera with {encoder_name}: "
            f"{self.segment_pattern.name} (starting segment {self._segment_start_number})",
            flush=True,
        )

    def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for raw_line in self._process.stderr:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                self._encoder_errors.append(line)

    def _report_encoder_failure(self) -> None:
        if self._encoder_failure_reported:
            return
        self._encoder_failure_reported = True
        detail = "\n".join(self._encoder_errors) or "no FFmpeg diagnostic was emitted"
        print(f"Global video encoder stopped; dropping subsequent frames:\n{detail}", flush=True)

    def submit(self, frame: np.ndarray) -> None:
        """Copy and enqueue an RGB frame without blocking camera capture."""

        if self._closed:
            return
        if self._process is None or self._process.poll() is not None:
            self.dropped_frames += 1
            self._report_encoder_failure()
            return
        image = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if image.shape != expected:
            raise ValueError(
                f"global video frame must have shape {expected}, got {image.shape}"
            )
        try:
            self._frames.put_nowait((np.ascontiguousarray(image).copy(), time.time_ns()))
        except queue.Full:
            self.dropped_frames += 1

    def _run(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            while True:
                queued = self._frames.get()
                if queued is None:
                    return
                frame, wall_time_ns = queued
                self._process.stdin.write(frame.tobytes())
                if self.written_frames % self.anchor_every_frames == 0:
                    assert self._anchor_log is not None
                    self._anchor_log.write(json.dumps({
                        "kind": "anchor",
                        "video_frame": self.written_frames,
                        "wall_time_ns": wall_time_ns,
                    }, separators=(",", ":")) + "\n")
                self.written_frames += 1
        except (BrokenPipeError, OSError, ValueError):
            self._report_encoder_failure()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer_thread = self._writer_thread
        process = self._process
        if writer_thread is not None:
            while True:
                try:
                    self._frames.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
            writer_thread.join(timeout=10.0)
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
        if self._anchor_log is not None:
            self._anchor_log.close()
        if process is not None and process.returncode not in (None, 0):
            self._report_encoder_failure()
        print(
            f"Global video segments saved as {self.segment_pattern} "
            f"({self.written_frames} frames, {self.dropped_frames} dropped; "
            f"anchors: {self.anchor_path})",
            flush=True,
        )
