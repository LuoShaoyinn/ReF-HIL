from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyspacemouse

from shared.zmq import Receiver, Sender, ZmqEndpointConfig

from .copilot import rotation_copilot_action
from .toggle import (
    resolve_gripper_command,
    spacemouse_gripper_buttons,
    spacemouse_has_intent,
)


class SpacemouseServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        request_port: int = 7002,
        response_host: str = "127.0.0.1",
        response_port: int = 7003,
        device: str | None = None,
        device_index: int = 0,
        override_rotation: bool = False,
    ) -> None:
        self.device = device
        self.device_index = int(device_index)
        self.override_rotation = bool(override_rotation)
        self.request_receiver = Receiver(ZmqEndpointConfig(host=host, port=request_port), timeout_ms=0)
        self.response_sender = Sender(ZmqEndpointConfig(host=response_host, port=response_port), timeout_ms=2000)
        open_kwargs: dict[str, Any] = {
            "nonblocking": True,
            "device_index": int(self.device_index),
        }
        if self.device is not None:
            open_kwargs["device"] = str(self.device)
        self.spacemouse_device = pyspacemouse.open(**open_kwargs)
        self._last_gripper_command = -1.0
        self._was_intervening = False
        self._previous_buttons = (False, False)
        self._warned_missing_copilot_context = False

    def serve_forever(self) -> None:
        try:
            mode = "ENABLED" if self.override_rotation else "disabled"
            print(
                f"Spacemouse Server Started (rotation override: {mode})",
                flush=True,
            )
            while True:
                state = self.spacemouse_device.read()
                request = self.request_receiver.recv(False)
                command = (
                    request.get("command") if isinstance(request, dict) else None
                )
                if command == "reset":
                    if self._was_intervening:
                        print(
                            "Human intervention ENDED (episode reset)",
                            flush=True,
                        )
                    self._was_intervening = False
                    self._previous_buttons = spacemouse_gripper_buttons(
                        state.buttons
                    )

                if command == "read_action":
                    current_gripper = request.get("gripper_position")
                    if current_gripper is not None:
                        current_gripper = float(
                            np.clip(current_gripper, 0.0, 1.0)
                        )
                    delta_pos = np.asarray(
                        [float(state.x), float(state.y), float(state.z)],
                        dtype=np.float32,
                    )
                    manual_delta_rot = np.asarray(
                        [float(state.roll), -float(state.pitch), float(state.yaw)],
                        dtype=np.float32,
                    )
                    delta_rot = manual_delta_rot
                    buttons = spacemouse_gripper_buttons(state.buttons)
                    close_pressed, open_pressed = buttons
                    gripper = resolve_gripper_command(
                        state.buttons,
                        current_gripper,
                        fallback=self._last_gripper_command,
                        last_command=self._last_gripper_command,
                    )
                    self._last_gripper_command = gripper
                    is_intervene = spacemouse_has_intent(
                        delta_pos, manual_delta_rot, state.buttons
                    )
                    manual_rotation_active = bool(
                        np.linalg.norm(manual_delta_rot) > 1e-3
                    )
                    copilot = request.get("rotation_copilot")
                    if self.override_rotation and isinstance(copilot, dict):
                        tcp_pose = np.asarray(request["tcp_pose"], dtype=np.float32).reshape(2, 3)
                        delta_rot = rotation_copilot_action(
                            tcp_pose[1],
                            np.asarray(copilot["target_rotvec"], dtype=np.float32),
                            angular_speed=float(copilot["angular_speed"]),
                            gain=float(copilot.get("gain", 1.0)),
                            deadband_rad=float(copilot.get("deadband_rad", 0.01)),
                        )
                    elif self.override_rotation and not self._warned_missing_copilot_context:
                        print(
                            "WARNING: rotation override requested, but this client "
                            "did not send TCP pose/copilot context"
                        )
                        self._warned_missing_copilot_context = True

                    previous_close, previous_open = self._previous_buttons
                    if open_pressed and not previous_open:
                        print("Gripper absolute command OPEN", flush=True)
                    elif close_pressed and not previous_close:
                        print("Gripper absolute command CLOSED", flush=True)
                    self._previous_buttons = buttons
                    if is_intervene != self._was_intervening:
                        event = "STARTED" if is_intervene else "ENDED"
                        print(f"Human intervention {event}", flush=True)
                        self._was_intervening = is_intervene

                    self.response_sender.send({
                        "delta_pos": delta_pos,
                        "delta_rot": delta_rot,
                        "manual_rotation_active": manual_rotation_active,
                        "gripper": gripper,
                        "gripper_position": current_gripper,
                        "gripper_close_pressed": close_pressed,
                        "gripper_open_pressed": open_pressed,
                        "gripper_pressed": close_pressed or open_pressed,
                        "is_intervene": is_intervene,
                    })
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("Spacemouse Server Stopped")
        finally:
            self.spacemouse_device.close()
            self.request_receiver.close()
            self.response_sender.close()
