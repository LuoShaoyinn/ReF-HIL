from __future__ import annotations

from abc import abstractmethod, ABC
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from shared.zmq import Receiver, Sender, ZmqEndpointConfig
from .realsense_camera import RealSenseCamera, RealSenseCameraConfig
from .robotiq_gripper import RobotiqGripper, RobotiqGripperConfig
from .ur_impedance_control import URImpedanceControl, URImpedanceControlConfig
from .global_video import GlobalVideoRecorder


@dataclass(kw_only=True)
class UR5eBaseConfig:
    request_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="0.0.0.0", port=7001)
    )
    response_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="127.0.0.1", port=7008)
    )
    fake_mode: bool = False
    include_raw_images: bool = False
    global_video_path: Path | None = None
    global_video_anchor_every_frames: int = 30
    control_log_path: Path | None = None
    realsense: dict[str, RealSenseCameraConfig] = field(
        default_factory=lambda: {
            "global":  RealSenseCameraConfig(serial_number="130322274099", use_depth=True),
            "wrist_0": RealSenseCameraConfig(serial_number="230422272349"),
            "wrist_1": RealSenseCameraConfig(serial_number="419122270589"),
        }
    )
    impendence_control: URImpedanceControlConfig = field(default_factory=URImpedanceControlConfig)
    gripper: RobotiqGripperConfig = field(default_factory=RobotiqGripperConfig)


class UR5eBase(ABC):
    def __init__(self, config: UR5eBaseConfig) -> None:
        self.config = config
        self.request_receiver = Receiver(self.config.request_endpoint, timeout_ms=0)
        self.response_sender = Sender(self.config.response_endpoint, timeout_ms=2000)

        self.impedance_control = URImpedanceControl(self.config.impendence_control)
        self.cameras = {name: RealSenseCamera(cfg) for name, cfg in self.config.realsense.items()}
        self.gripper = RobotiqGripper(self.config.gripper)

        self.camera_thread = threading.Thread(target=self.read_cameras, daemon=True)
        self.global_video: GlobalVideoRecorder | None = None

    def connect(self) -> None:
        self.impedance_control.connect()
        for camera in self.cameras.values():
            camera.connect()
        self.gripper.connect()

        if self.config.global_video_path is not None:
            global_camera = self.cameras["global"]
            self.global_video = GlobalVideoRecorder(
                self.config.global_video_path,
                width=global_camera.config.width,
                height=global_camera.config.height,
                fps=global_camera.config.fps,
                anchor_every_frames=self.config.global_video_anchor_every_frames,
            )
            self.global_video.start()
            print(
                f"Recording global camera with hardware FFmpeg to "
                f"{self.config.global_video_path}",
                flush=True,
            )

        self.camera_thread.start()

    def run(self) -> None:
        self.connect()
        control_log = None
        if self.config.control_log_path is not None:
            path = Path(self.config.control_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            control_log = path.open("w", encoding="utf-8", buffering=1024 * 1024)
            print(f"Writing controller diagnostics to {path}", flush=True)
        logged_samples = 0
        try:
            print("START")
            while True:
                request = self.request_receiver.recv(None)
                if isinstance(request, dict):
                    self.apply_action(request)
                elif request is True:
                    self.response_sender.send(self.read_obs())
                elif request == "read_global_camera":
                    self.response_sender.send(self.cameras["global"].read_image())
                elif request == "read_wrist_0_camera":
                    self.response_sender.send(self.cameras["wrist_0"].read_image())
                elif request == "read_wrist_1_camera":
                    self.response_sender.send(self.cameras["wrist_1"].read_image())
                elif request == "read_tcp_pose":
                    self.response_sender.send(self.impedance_control.get_actual_tcp_pose().astype(np.float32))
                diagnostic = self.impedance_control.pid_step(
                    diagnostics=control_log is not None
                )
                if control_log is not None and diagnostic is not None:
                    serializable = {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in diagnostic.items()
                    }
                    control_log.write(
                        json.dumps(serializable, separators=(",", ":")) + "\n"
                    )
                    logged_samples += 1
                    if logged_samples % 500 == 0:
                        control_log.flush()
                time.sleep(0.002)
        except KeyboardInterrupt as e:
            print("STOP")
        finally:
            if self.global_video is not None:
                self.global_video.close()
            if control_log is not None:
                control_log.close()
            self.gripper.disconnect()
            for camera in self.cameras.values():
                camera.disconnect()
            self.impedance_control.disconnect()
            self.request_receiver.close()
            self.response_sender.close()

    def read_cameras(self) -> None:
        try:
            while True:
                for name, camera in self.cameras.items():
                    camera.update()
                    if name == "global" and self.global_video is not None:
                        self.global_video.submit(camera.read_image())
        except RuntimeError:
            pass  # camera is not connected, just exit the thread
        finally:
            print("Camera thread exiting")

    @abstractmethod
    def apply_action(self, raw_action: dict) -> None:
        pass

    @abstractmethod
    def read_obs(self) -> dict:
        pass
