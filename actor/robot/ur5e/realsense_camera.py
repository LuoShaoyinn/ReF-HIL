from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import pyrealsense2 as rs # type: ignore[import]


@dataclass(frozen=True, kw_only=True)
class RealSenseCameraConfig:
    serial_number: str = ""
    fps: int = 30
    width: int = 640
    height: int = 480
    depth_min: float = 2000
    depth_max: float = 4000
    use_depth: bool = False


class RealSenseCamera:
    def __init__(self, config: RealSenseCameraConfig) -> None:
        self.config = config
        self.pipeline: rs.pipeline | None = None
        self.frame = None
        self.align_to_color = None

    @property
    def use_depth(self) -> bool:
        return bool(self.config.use_depth)

    def connect(self) -> None:
        if self.pipeline is not None:
            return
        self.pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(self.config.serial_number)
        stream_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps,
        )
        if self.use_depth:
            stream_config.enable_stream(
                rs.stream.depth,
                self.config.width,
                self.config.height,
                rs.format.z16,
                self.config.fps,
            )
        try:
            self.pipeline.start(stream_config)
            if self.use_depth:
                self.align_to_color = rs.align(rs.stream.color)
            # The observation server may read immediately after connect().
            # Do not expose a connected camera until its first frame is ready.
            self.update()
        except Exception:
            try:
                self.disconnect()
            except RuntimeError:
                pass  # start() may have failed before the pipeline was running.
            raise

    def disconnect(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.align_to_color = None
        self.frame = None
        if pipeline is not None:
            pipeline.stop()

    def update(self):
        assert self.pipeline is not None, "camera is not connected"
        frame = self.pipeline.wait_for_frames(timeout_ms=5000)
        if self.use_depth:
            frame = self.align_to_color.process(frame)
        if not frame.get_color_frame() or (self.use_depth and not frame.get_depth_frame()):
            raise RuntimeError(f"RealSense {self.config.serial_number}: incomplete camera frame")
        self.frame = frame

    def read_image(self) -> np.ndarray:
        if self.frame is None:
            raise RuntimeError("camera has no frame; connect() must complete before reading")
        color = self.frame.get_color_frame()
        return np.asarray(color.get_data(), dtype=np.uint8)

    def read_depth(self) -> np.ndarray:
        if not self.use_depth or self.frame is None:
            raise RuntimeError("depth camera is not ready")
        depth = self.frame.get_depth_frame()
        scale = self.config.depth_max - self.config.depth_min
        depth = np.asarray(depth.get_data(), dtype=np.float32) - self.config.depth_min
        depth = np.clip(depth / scale * 255.0, 0, 255.0)
        return depth.astype(np.uint8)
