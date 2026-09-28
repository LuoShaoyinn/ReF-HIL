from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Any

import zmq

DictMessage = dict[str, Any]


@dataclass(kw_only=True)
class ZmqEndpointConfig:
    host: str
    port: int

    @property
    def uri(self) -> str:
        return f"tcp://{self.host}:{self.port}"


class Sender:
    def __init__(self, endpoint: ZmqEndpointConfig, timeout_ms: int = 2000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint.uri)

    def send(self, data: object) -> None:
        assert self._socket.poll(self.timeout_ms, zmq.POLLOUT) != 0, f"send timeout to {self.endpoint.uri}"
        self._socket.send(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))

    def close(self) -> None:
        self._socket.close(0)

    def __del__(self) -> None:
        self.close()

class Receiver:
    def __init__(self, endpoint: ZmqEndpointConfig, timeout_ms: int = 0) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self.endpoint.uri)

    def recv(self, default: Any) -> Any:
        if self._socket.poll(self.timeout_ms, zmq.POLLIN) == 0:
            return default
        data = self._socket.recv()
        return pickle.loads(data)

    def close(self) -> None:
        self._socket.close(0)

    def __del__(self) -> None:
        self.close()
