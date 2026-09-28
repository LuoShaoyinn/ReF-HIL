from __future__ import annotations

from dataclasses import dataclass, field

from shared.zmq import Receiver, Sender, ZmqEndpointConfig
from .base import BaseClassifier, BaseClassifierConfig


@dataclass(kw_only=True)
class HumanClassifierConfig(BaseClassifierConfig):
    gui_key_request_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="127.0.0.1", port=7006)
    )
    gui_key_response_endpoint: ZmqEndpointConfig = field(
        default_factory=lambda: ZmqEndpointConfig(host="0.0.0.0", port=7007)
    )
    success_key: int = 32  # glfw.KEY_SPACE
    failed_key: int = 70   # glfw.KEY_F


class HumanClassifier(BaseClassifier):
    def __init__(self, config: HumanClassifierConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.key_sender = Sender(self.config.gui_key_request_endpoint)
        self.key_receiver = Receiver(self.config.gui_key_response_endpoint, timeout_ms=100)

    def inference(self, raw_robot_observation: dict) -> bool:
        _ = raw_robot_observation
        self.key_sender.send(int(self.config.failed_key))
        if bool(self.key_receiver.recv(False)):
            return False
        self.key_sender.send(int(self.config.success_key))
        return bool(self.key_receiver.recv(False))
