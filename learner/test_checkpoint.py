from __future__ import annotations

import pickle
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from learner.checkpoint import (
    CheckpointWriter,
    immutable_cpu_copy,
    _pack_tensors,
    _unpack_tensors,
)
from learner.runtime import Learner, LearnerConfig
from shared.policy import BasePolicy, PolicyConfig
from shared.zmq import ZmqEndpointConfig


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _stress_checkpoint_transport(directory: str) -> None:
    """Run in a separate process so the test suite's limits stay untouched."""
    import resource

    resource.setrlimit(
        resource.RLIMIT_NOFILE, (256, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    )
    writer = CheckpointWriter(asynchronous=True)
    rows = []
    try:
        for index in range(8):
            # More independent storages than the FD limit, on EVERY save.
            state = {
                "state": [torch.tensor([float(i), float(index)]) for i in range(1200)]
            }
            path = Path(directory) / f"checkpoint_{index}.pkl"
            writer.submit(path, immutable_cpu_copy(state), wait=True)
            with path.open("rb") as handle:
                restored = pickle.load(handle)
            torch.testing.assert_close(restored["state"][999], state["state"][999])
            rows.append(
                (
                    len(os.listdir("/proc/self/fd")),
                    writer._ensure_executor()
                    .submit(_worker_fd_count)
                    .result(timeout=20),
                )
            )
    finally:
        writer.close()
    with (Path(directory) / "fd_counts.json").open("w") as handle:
        json.dump(rows, handle)


class _Task:
    class _Config:
        obs_dims = 1
        action_dims = 1

    config = _Config()

    @staticmethod
    def build_obs(state, _info, augment=False):
        del augment
        return torch.tensor([state["x"]], dtype=torch.float32)

    @staticmethod
    def build_action(action, _info, augment=False):
        del augment
        return torch.tensor([action["x"]], dtype=torch.float32)


class _Policy(BasePolicy):
    def __init__(self) -> None:
        super().__init__(config=PolicyConfig(device="cpu"))
        self.device = torch.device("cpu")
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def sample_action(self, observation, deterministic=False):
        del deterministic
        return self.model(observation)

    def update(self, batch):
        del batch
        return {}

    def export(self):
        return {
            "model": {
                key: value.detach().cpu().numpy()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
        }

    def load(self, state):
        self.model.load_state_dict(
            {key: torch.as_tensor(value) for key, value in state["model"].items()}
        )
        self.optimizer.load_state_dict(state["optimizer"])


class CheckpointTest(unittest.TestCase):
    def test_background_write_failure_closes_executor_and_removes_temp(self):
        with tempfile.TemporaryDirectory(prefix="srt-checkpoint-failure-") as tmp:
            destination = Path(tmp) / "existing_directory"
            destination.mkdir()
            writer = CheckpointWriter(asynchronous=True)
            writer.submit(destination, {"tensor": torch.ones(2)}, wait=False)
            with self.assertRaises(OSError):
                writer.close()
            self.assertIsNone(writer._executor)
            self.assertIsNone(writer._pending)
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_transport_preserves_types_shapes_and_dtypes(self) -> None:
        tensors = [
            torch.tensor(2.0),
            torch.arange(6).reshape(2, 3).T,
            torch.empty(0, 3),
            torch.tensor([True, False]),
            torch.ones(3, dtype=torch.bfloat16),
            torch.tensor([1 + 2j]).conj(),
            torch.ones(3, dtype=torch.float16),
        ]
        source = {
            "tensors": tensors,
            "nested": ({"value": torch.tensor(3)},),
            "array": np.asarray([1.0, 2.0]),
            "number": 4,
        }
        packed = _pack_tensors(source)

        def verify_no_tensors(value):
            self.assertNotIsInstance(value, torch.Tensor)
            if isinstance(value, dict):
                for v in value.values():
                    verify_no_tensors(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    verify_no_tensors(v)
            elif hasattr(value, "data") and not isinstance(value, np.ndarray):
                self.assertIsInstance(value.data, np.ndarray)

        verify_no_tensors(packed)
        source["nested"][0]["value"].add_(1)
        restored = _unpack_tensors(packed)
        for a, b in zip(tensors, restored["tensors"]):
            torch.testing.assert_close(a, b)
        self.assertEqual(restored["nested"][0]["value"].item(), 3)
        self.assertIsInstance(restored["array"], np.ndarray)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux FD accounting")
    def test_repeated_background_saves_do_not_exhaust_or_leak_fds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-checkpoint-fds-") as tmp:
            code = (
                "from learner.test_checkpoint import _stress_checkpoint_transport; "
                f"_stress_checkpoint_transport({tmp!r})"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=90
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (Path(tmp) / "fd_counts.json").open() as handle:
                rows = json.load(handle)
            print(f"Checkpoint FD counts (learner, writer): {rows}", flush=True)
            self.assertEqual(len(rows), 8)
            for column in (0, 1):
                counts = [r[column] for r in rows]
                self.assertLessEqual(max(counts) - min(counts), 2, rows)
                self.assertLess(max(counts), 64, rows)
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_async_optimizer_resume_matches_next_update(self) -> None:
        torch.manual_seed(19)
        original = _Policy()

        def step(policy):
            policy.optimizer.zero_grad()
            policy.model(torch.ones(3, 1)).square().mean().backward()
            policy.optimizer.step()

        step(original)
        with tempfile.TemporaryDirectory(prefix="srt-optimizer-async-") as tmp:
            path = Path(tmp) / "checkpoint.pkl"
            writer = CheckpointWriter(asynchronous=True)
            try:
                writer.submit(path, immutable_cpu_copy(original.export()), wait=True)
            finally:
                writer.close()
            with path.open("rb") as handle:
                state = pickle.load(handle)
        resumed = _Policy()
        resumed.load(state)
        self.assertIsInstance(
            next(iter(state["optimizer"]["state"].values()))["exp_avg"], torch.Tensor
        )
        step(original)
        step(resumed)
        for a, b in zip(original.model.parameters(), resumed.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_immutable_copy_detaches_live_tensors(self) -> None:
        source = torch.tensor([1.0])
        copied = immutable_cpu_copy({"value": source})["value"]
        source.add_(2.0)
        torch.testing.assert_close(copied, torch.tensor([1.0]))

    def test_background_writer_publishes_one_atomic_pickle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-checkpoint-writer-") as tmp:
            path = Path(tmp) / "checkpoint_00000010s.pkl"
            writer = CheckpointWriter(asynchronous=True)
            self.assertTrue(writer.submit(path, {"value": 7}, wait=False))
            writer.close()
            with path.open("rb") as handle:
                self.assertEqual(pickle.load(handle), {"value": 7})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_one_file_round_trip_restores_optimizer_and_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-checkpoint-roundtrip-") as tmp:
            root = Path(tmp)
            config = LearnerConfig(
                transitions_endpoint=ZmqEndpointConfig(
                    host="127.0.0.1", port=_free_port()
                ),
                actor_parameters_endpoint=ZmqEndpointConfig(
                    host="127.0.0.1", port=_free_port()
                ),
                output_root=root,
                experiment_name="run",
                replay_device="cpu",
                max_buffer_size=8,
                save_every_seconds=0,
            )
            source_policy = _Policy()
            source = Learner(config=config, task=_Task(), policy=source_policy)
            try:
                loss = source_policy.model(torch.ones((1, 1))).square().mean()
                source_policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                source_policy.optimizer.step()
                source.train_steps = 17
                source.actor_steps = 23
                source.actor_episodes = 5
                source.online_buffer.add(
                    {
                        "raw_obs": {"x": 3.0},
                        "raw_action": {"x": -0.5},
                        "reward": 1.0,
                        "done": True,
                        "info": {"is_intervene": False},
                    }
                )
                source._run_started_monotonic = time.monotonic() - 125.9
                expected_weight = source_policy.model.weight.detach().clone()
                expected_optimizer_step = (
                    next(iter(source_policy.optimizer.state.values()))["step"]
                    .detach()
                    .clone()
                )
                source._save_model_and_state()
            finally:
                source._checkpoint_writer.close()
                source.transition_receiver.close()
                source.parameters_sender.close()
                source.writer.close()

            paths = list((root / "run" / "checkpoints").glob("*.pkl"))
            self.assertEqual(
                [path.name for path in paths], ["checkpoint_00000125s.pkl"]
            )
            with paths[0].open("rb") as handle:
                payload = pickle.load(handle)
            self.assertNotIn("buffer", payload)
            self.assertEqual(payload["class_name"], "Learner")
            self.assertIn("actor", payload)
            self.assertEqual(payload["replay_watermark"], 1)
            self.assertTrue((root / "run" / "buffer" / "0000.pkl").is_file())

            resumed_policy = _Policy()
            resumed = Learner(config=config, task=_Task(), policy=resumed_policy)
            try:
                self.assertEqual(resumed.train_steps, 17)
                self.assertEqual(resumed.actor_steps, 23)
                self.assertEqual(resumed.actor_episodes, 5)
                self.assertEqual(resumed.training_elapsed_seconds, 125.0)
                self.assertEqual(resumed.online_buffer.raw_count, 1)
                torch.testing.assert_close(resumed_policy.model.weight, expected_weight)
                restored_step = next(iter(resumed_policy.optimizer.state.values()))[
                    "step"
                ]
                torch.testing.assert_close(restored_step, expected_optimizer_step)
            finally:
                resumed._checkpoint_writer.close()
                resumed.transition_receiver.close()
                resumed.parameters_sender.close()
                resumed.writer.close()


if __name__ == "__main__":
    unittest.main()
