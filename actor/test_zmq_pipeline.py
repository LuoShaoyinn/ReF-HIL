from __future__ import annotations

import socket
import unittest

import torch
import numpy as np

from actor.runtime import Actor, ActorConfig
from actor.transport import FakeOperatorClient, FakeRobotClient
from shared.zmq import Receiver, ZmqEndpointConfig


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Task:
    def reset(self, _send_action: object, _read_observation: object) -> None:
        pass

    def prepare_observation(self, raw: dict) -> dict:
        return {
            "tcp_speed": np.zeros(6, dtype=np.float32),
            "tcp_force": np.zeros(6, dtype=np.float32),
            "gripper": np.float32(-1.0),
            "projected_gravity": np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
            "images": raw["images"],
        }

    def build_obs(self, _raw: dict, _info: dict) -> torch.Tensor:
        return torch.zeros((1,))

    def build_operator_request(self, _raw: dict) -> dict:
        return {}

    def select_action(self, action: torch.Tensor, _human: dict, _last: float) -> tuple[torch.Tensor, bool]:
        return action, False

    def parse_action(self, _action: torch.Tensor) -> dict:
        return {"delta_pos": [0.0, 0.0, 0.0], "gripper": -1.0}

    def build_info(self, **kwargs: object) -> dict:
        return dict(kwargs)


class _Policy:
    def sample_action(self, _obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1,))

    def load(self, _state: dict) -> None:
        pass


class ZmqPipelineTest(unittest.TestCase):
    def test_fake_hardware_delivers_complete_episode(self) -> None:
        episode_port = _free_port()
        parameter_port = _free_port()
        receiver = Receiver(ZmqEndpointConfig(host="127.0.0.1", port=episode_port), timeout_ms=2000)
        actor = Actor(
            config=ActorConfig(
                learner_endpoint=ZmqEndpointConfig(host="127.0.0.1", port=episode_port),
                parameters_endpoint=ZmqEndpointConfig(host="127.0.0.1", port=parameter_port),
                max_steps_per_episode=1,
                loop_sleep_s=0.0,
                wait_for_initial_parameters=False,
            ),
            task=_Task(),
            policy=_Policy(),
            robot=FakeRobotClient(
                image_size=8,
                linear_speed=0.05,
                initial_position=(-0.250, -0.580, 0.060),
                initial_rotation=(2.221, 2.221, 0.0),
            ),
            operator=FakeOperatorClient(),
            success_oracle=lambda _raw: True,
        )
        try:
            actor.run_episode()
            episode = receiver.recv(None)
        finally:
            actor.learner_sender.close()
            actor.parameters_receiver.close()
            receiver.close()

        self.assertIsInstance(episode, list)
        self.assertEqual(len(episode), 1)
        self.assertTrue(episode[-1]["done"])
        self.assertIn("raw_obs", episode[0])


if __name__ == "__main__":
    unittest.main()
