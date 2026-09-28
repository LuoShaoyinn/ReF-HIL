from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

from actor.runtime import Actor, ActorConfig
from actor.terminal_success import TerminalSuccessKey


class TerminalSuccessTest(unittest.TestCase):
    def test_reader_consumes_all_chunks_after_enter(self) -> None:
        key = TerminalSuccessKey()
        key._fd = 123
        with patch("actor.terminal_success.select.select", side_effect=[([123], [], []), ([123], [], []), ([], [], [])]), patch("actor.terminal_success.os.read", side_effect=[b"\n", b"\r\n"]) as read:
            self.assertTrue(key.requested())
            self.assertEqual(read.call_count, 2)
        with patch("actor.terminal_success.termios.tcflush") as flush:
            key.drain()
            flush.assert_called_once()


class _Socket:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.sent: list[object] = []
        self.timeout_ms = 0

    def send(self, value: object) -> None:
        self.sent.append(value)

    def recv(self, default: object) -> object:
        return default

    def close(self) -> None:
        pass


class _Robot:
    def __init__(self) -> None:
        self.actions: list[dict] = []
        self.observations = [{"frame": 0}, {"frame": 1}]

    def read_observation(self) -> dict:
        return self.observations.pop(0)

    def send_action(self, action: dict) -> None:
        self.actions.append(action)

    def close(self) -> None:
        pass


class _Operator:
    def __init__(self) -> None:
        self.reset_count = 0
        self.requests: list[dict | None] = []

    def reset(self) -> None:
        self.reset_count += 1

    def read_action(self, request: dict | None = None) -> dict:
        self.requests.append(request)
        return {"human": False}

    def close(self) -> None:
        pass


class _Task:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self, send_action: object, read_observation: object) -> None:
        self.reset_count += 1

    def prepare_observation(self, raw: dict) -> dict:
        return {
            **raw,
            "tcp_speed": np.zeros(6, dtype=np.float32),
            "tcp_force": np.zeros(6, dtype=np.float32),
            "gripper": np.float32(-1.0),
            "projected_gravity": np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        }

    def build_obs(self, raw: dict, _info: dict) -> torch.Tensor:
        return torch.tensor([float(raw["frame"])])

    def build_operator_request(self, raw: dict) -> dict:
        return {"frame": raw["frame"]}

    def select_action(self, action: torch.Tensor, _human: dict, _last_gripper: float) -> tuple[torch.Tensor, bool]:
        return action, False

    def parse_action(self, action: torch.Tensor) -> dict:
        return {"gripper": float(action[0]), "delta_pos": [0.0, 0.0, 0.0]}

    def build_info(self, **kwargs: object) -> dict:
        return dict(kwargs)


class _Policy:
    def sample_action(self, _obs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([1.0])

    def load(self, _state: dict) -> None:
        pass


class _InterveningTask(_Task):
    def select_action(
        self,
        action: torch.Tensor,
        _human: dict,
        _last_gripper: float,
    ) -> tuple[torch.Tensor, bool]:
        return -action, True


class RuntimeActorTest(unittest.TestCase):
    def test_actor_emits_raw_episode_without_gui(self) -> None:
        sender_instances: list[_Socket] = []

        def sender_factory(*args: object, **kwargs: object) -> _Socket:
            sender = _Socket(*args, **kwargs)
            sender_instances.append(sender)
            return sender

        with patch("actor.runtime.Sender", side_effect=sender_factory), patch("actor.runtime.Receiver", _Socket):
            actor = Actor(
                config=ActorConfig(max_steps_per_episode=1, loop_sleep_s=0.0),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                operator=_Operator(),
                success_oracle=lambda raw: raw["frame"] == 1,
            )
            episode = actor.run_episode()

        self.assertEqual(len(episode), 1)
        self.assertTrue(episode[0]["done"])
        self.assertEqual(episode[0]["reward"], 1.0)
        self.assertNotIn("gui", actor.__dict__)
        self.assertEqual(sender_instances[0].sent, [episode])
        self.assertEqual(actor.operator.requests, [{"frame": 0}])

    def test_intervention_preserves_same_state_autonomous_proposal(self) -> None:
        with patch("actor.runtime.Sender", _Socket), patch(
            "actor.runtime.Receiver", _Socket
        ):
            actor = Actor(
                config=ActorConfig(max_steps_per_episode=1, loop_sleep_s=0.0),
                task=_InterveningTask(),
                policy=_Policy(),
                robot=_Robot(),
                operator=_Operator(),
                success_oracle=lambda raw: raw["frame"] == 1,
            )
            episode = actor.run_episode()

        self.assertEqual(episode[0]["raw_action"]["gripper"], -1.0)
        self.assertEqual(
            episode[0]["info"]["autonomous_action"]["gripper"], 1.0
        )

    def test_enter_override_marks_episode_success(self) -> None:
        with patch("actor.runtime.Sender", _Socket), patch(
            "actor.runtime.Receiver", _Socket
        ):
            actor = Actor(
                config=ActorConfig(max_steps_per_episode=1, loop_sleep_s=0.0),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                operator=_Operator(),
                success_oracle=lambda _raw: False,
                force_success_requested=lambda: True,
            )
            episode = actor.run_episode()

        self.assertEqual(len(episode), 1)
        self.assertTrue(episode[0]["done"])
        self.assertEqual(episode[0]["reward"], 1.0)
        self.assertTrue(episode[0]["info"]["forced_success"])

    def test_episode_start_discards_stale_success_request(self) -> None:
        pending = [True]
        with patch("actor.runtime.Sender", _Socket), patch("actor.runtime.Receiver", _Socket):
            actor = Actor(
                config=ActorConfig(max_steps_per_episode=1, loop_sleep_s=0.0),
                task=_Task(), policy=_Policy(), robot=_Robot(), operator=_Operator(),
                success_oracle=lambda _raw: False,
                force_success_requested=lambda: bool(pending),
                drain_force_success=pending.clear,
            )
            episode = actor.run_episode()
        self.assertLessEqual(episode[-1]["reward"], 0.0)
        self.assertFalse(episode[-1]["info"].get("forced_success", False))

    def test_enter_between_steps_finishes_previous_transition(self) -> None:
        requests = iter((False, False, True))
        robot = _Robot()
        with patch("actor.runtime.Sender", _Socket), patch(
            "actor.runtime.Receiver", _Socket
        ):
            actor = Actor(
                config=ActorConfig(max_steps_per_episode=2, loop_sleep_s=0.0),
                task=_Task(),
                policy=_Policy(),
                robot=robot,
                operator=_Operator(),
                success_oracle=lambda _raw: False,
                force_success_requested=lambda: next(requests),
            )
            episode = actor.run_episode()

        self.assertEqual(len(episode), 1)
        self.assertEqual(len(robot.actions), 1)
        self.assertTrue(episode[0]["done"])
        self.assertEqual(episode[0]["reward"], 1.0)
        self.assertTrue(episode[0]["info"]["forced_success"])

    def test_actor_run_renders_terminal_episode_status(self) -> None:
        with patch("actor.runtime.Sender", _Socket), patch(
            "actor.runtime.Receiver", _Socket
        ):
            actor = Actor(
                config=ActorConfig(
                    max_episodes=1,
                    max_steps_per_episode=1,
                    loop_sleep_s=0.0,
                ),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                operator=_Operator(),
                success_oracle=lambda raw: raw["frame"] == 1,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                actor.run()

        self.assertIn("[1]", output.getvalue())
        self.assertIn("success=True", output.getvalue())


if __name__ == "__main__":
    unittest.main()
