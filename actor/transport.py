"""Transport clients used by the actor runtime.

The robot driver and the SpaceMouse driver remain independent processes. This
module owns only their wire protocol, so task/algorithm code never constructs
ZMQ sockets for hardware I/O.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from shared.zmq import DictMessage, Receiver, Sender, ZmqEndpointConfig


class RobotClient(Protocol):
    def read_observation(self) -> DictMessage: ...

    def send_action(self, action: DictMessage) -> None: ...

    def close(self) -> None: ...


class OperatorClient(Protocol):
    def reset(self) -> None: ...

    def read_action(self, request: DictMessage | None = None) -> DictMessage: ...

    def close(self) -> None: ...


class ZmqRobotClient:
    """Client for the task-owned UR5e driver request/observation protocol."""

    def __init__(
        self,
        *,
        request_endpoint: ZmqEndpointConfig,
        response_endpoint: ZmqEndpointConfig,
        timeout_ms: int = 2000,
    ) -> None:
        self._request = Sender(request_endpoint)
        self._response = Receiver(response_endpoint, timeout_ms=timeout_ms)

    def read_observation(self) -> DictMessage:
        self._request.send(True)
        return self._response.recv({})

    def send_action(self, action: DictMessage) -> None:
        self._request.send(action)

    def close(self) -> None:
        self._request.close()
        self._response.close()


class ZmqOperatorClient:
    """Client for the independently-running SpaceMouse driver."""

    def __init__(
        self,
        *,
        request_endpoint: ZmqEndpointConfig,
        response_endpoint: ZmqEndpointConfig,
        timeout_ms: int = 2000,
    ) -> None:
        self._request = Sender(request_endpoint)
        self._response = Receiver(response_endpoint, timeout_ms=timeout_ms)

    def reset(self) -> None:
        self._request.send({"command": "reset"})

    def read_action(self, request: DictMessage | None = None) -> DictMessage:
        message = {"command": "read_action"}
        if request is not None:
            message.update(request)
        self._request.send(message)
        return self._response.recv({})

    def close(self) -> None:
        self._request.close()
        self._response.close()


class FakeRobotClient:
    """Deterministic hardware-free robot source for actor/learner smoke tests."""

    def __init__(
        self,
        *,
        image_size: int,
        linear_speed: float,
        initial_position: tuple[float, float, float],
        initial_rotation: tuple[float, float, float],
        gripper_position_range: tuple[float, float] = (0.0, 255.0),
    ) -> None:
        self._image_size = int(image_size)
        self._linear_speed = float(linear_speed)
        self._initial_rotation = np.asarray(initial_rotation, dtype=np.float32)
        self._gripper_position_range = tuple(map(float, gripper_position_range))
        self._step = 0
        self._gripper = -1.0
        self._tcp_pos = np.asarray(initial_position, dtype=np.float32)

    def send_action(self, action: DictMessage) -> None:
        self._tcp_pos += (
            np.clip(
                np.asarray(
                    action.get("delta_pos", [0.0, 0.0, 0.0]), dtype=np.float32
                ),
                -1.0,
                1.0,
            )
            * self._linear_speed
        )
        self._gripper = float(np.clip(action.get("gripper", self._gripper), -1.0, 1.0))
        self._step += 1

    def read_observation(self) -> DictMessage:
        pixel = np.uint8(min(255, self._step))
        image = np.full((self._image_size, self._image_size, 3), pixel, dtype=np.uint8)
        return {
            "tcp_pose": np.asarray([self._tcp_pos, self._initial_rotation], dtype=np.float32),
            "tcp_speed": np.zeros((2, 3), dtype=np.float32),
            "tcp_force": np.zeros((2, 3), dtype=np.float32),
            "gripper": np.asarray(
                self._gripper_position_range[0]
                + 0.5
                * (self._gripper + 1.0)
                * (self._gripper_position_range[1] - self._gripper_position_range[0]),
                dtype=np.float32,
            ),
            "images": {"global_0": image, "wrist_0": image, "wrist_1": image},
            "gripper_action_range": self._gripper_position_range,
        }

    def close(self) -> None:
        pass


class FakeOperatorClient:
    """No-intervention SpaceMouse substitute for deterministic smoke tests."""

    def reset(self) -> None:
        pass

    def read_action(self, request: DictMessage | None = None) -> DictMessage:
        del request
        return {
            "delta_pos": np.zeros((3,), dtype=np.float32),
            "delta_rot": np.zeros((3,), dtype=np.float32),
            "manual_rotation_active": False,
            "gripper": -1.0,
            "gripper_position": None,
            "gripper_close_pressed": False,
            "gripper_open_pressed": False,
            "gripper_pressed": False,
            "is_intervene": False,
        }

    def close(self) -> None:
        pass
