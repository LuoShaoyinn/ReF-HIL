from __future__ import annotations

import unittest

import numpy as np

from actor.transport import ZmqOperatorClient


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, value: dict) -> None:
        self.sent.append(value)


class _Receiver:
    def recv(self, default: dict) -> dict:
        del default
        return {"delta_rot": np.zeros(3, dtype=np.float32)}


class OperatorTransportTest(unittest.TestCase):
    def test_current_pose_and_copilot_config_are_sent_over_request_zmq(self) -> None:
        client = object.__new__(ZmqOperatorClient)
        client._request = _Sender()
        client._response = _Receiver()
        pose = np.asarray(
            [[-0.25, -0.58, 0.06], [2.221, 2.221, 0.0]], dtype=np.float32
        )
        context = {
            "tcp_pose": pose,
            "rotation_copilot": {"angular_speed": 0.1},
        }

        client.read_action(context)

        request = client._request.sent[0]
        self.assertEqual(request["command"], "read_action")
        np.testing.assert_array_equal(request["tcp_pose"], pose)
        self.assertEqual(request["rotation_copilot"]["angular_speed"], 0.1)


if __name__ == "__main__":
    unittest.main()
