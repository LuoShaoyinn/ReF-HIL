from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from learner.replay import DataBuffer, DataBufferConfig
from learner.subset_replay import RawIndexBuffer


class BatchedReplayTest(unittest.TestCase):
    def test_raw_only_add_returns_a_stable_sentinel(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-raw-replay-") as tmp:
            buffer = DataBuffer(
                DataBufferConfig(
                    capacity=8,
                    obs_dims=1,
                    action_dims=1,
                    autosave_dir=Path(tmp),
                    need_sample=False,
                )
            )
            transition = {
                "raw_obs": {},
                "raw_action": {},
                "reward": 0.0,
                "done": False,
                "info": {},
            }
            self.assertEqual(buffer.add(transition), -1)
            self.assertEqual(buffer.buffer, [transition])

    def test_episode_ingest_encodes_each_transition_once(self) -> None:
        calls: list[tuple[int, bool]] = []

        def build_observations(states: list[dict], _infos: list[dict], *, augment: bool) -> torch.Tensor:
            calls.append((len(states), augment))
            return torch.tensor([[state["id"] + (10 if augment else 0)] for state in states], dtype=torch.float32)

        def build_actions(actions: list[dict], _infos: list[dict], *, augment: bool) -> torch.Tensor:
            del augment
            return torch.tensor([[action["id"]] for action in actions], dtype=torch.float32)

        transitions = [
            {
                "raw_obs": {"id": index},
                "raw_action": {"id": 100 + index},
                "reward": float(index),
                "done": False,
                "info": {"is_intervene": False},
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory(prefix="srt-batched-replay-") as tmp:
            buffer = DataBuffer(
                DataBufferConfig(
                    capacity=8,
                    obs_dims=1,
                    action_dims=1,
                    info_features={"is_intervene": (1,)},
                    build_observations=build_observations,
                    build_actions=build_actions,
                    autosave_dir=Path(tmp),
                    data_device="cpu",
                    need_sample=True,
                )
            )
            final_indices = buffer.add_many(transitions)

        self.assertEqual(calls, [(3, False)])
        self.assertEqual(final_indices, [0, 1, 2])
        self.assertEqual(buffer.tot_transition, 3)
        self.assertEqual(buffer.data_obs[:3].reshape(-1).tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(buffer.data_action[:3].reshape(-1).tolist(), [100.0, 101.0, 102.0])

    def test_startup_recovery_encodes_each_chunk_as_one_batch(self) -> None:
        calls: list[int] = []

        def build_observations(states, _infos, *, augment):
            self.assertFalse(augment)
            calls.append(len(states))
            return torch.tensor([[row["id"]] for row in states]).float()

        def build_actions(actions, _infos, *, augment):
            self.assertFalse(augment)
            return torch.tensor([[row["id"]] for row in actions]).float()

        rows = [
            {
                "raw_obs": {"id": index},
                "raw_action": {"id": index},
                "reward": 0.0,
                "done": index == 2,
                "info": {"is_intervene": False},
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory(prefix="srt-replay-recovery-") as tmp:
            path = Path(tmp) / "0000.pkl"
            with path.open("wb") as handle:
                pickle.dump(rows, handle)
            replay = DataBuffer(
                DataBufferConfig(
                    capacity=8,
                    obs_dims=1,
                    action_dims=1,
                    info_features={"is_intervene": (1,)},
                    build_observations=build_observations,
                    build_actions=build_actions,
                    autosave_dir=tmp,
                    data_device="cpu",
                    need_sample=True,
                )
            )

        self.assertEqual(calls, [3])
        self.assertEqual(replay.tot_transition, 3)

    def test_raw_index_views_sample_grouped_views_and_same_next_raw_state(self) -> None:
        def observations(states, _infos, *, augment):
            offset = 10 if augment else 0
            return torch.tensor([[state["id"] + offset] for state in states]).float()

        def actions(rows, _infos, *, augment):
            del augment
            return torch.tensor([[row["id"]] for row in rows]).float()

        rows = [
            {
                "raw_obs": {"id": i},
                "raw_action": {"id": i + 20},
                "reward": 0.0,
                "done": i == 2,
                "info": {"is_intervene": i == 1},
            }
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory(prefix="srt-raw-view-") as tmp:
            replay = DataBuffer(
                DataBufferConfig(
                    capacity=8,
                    obs_dims=1,
                    action_dims=1,
                    info_features={"is_intervene": (1,)},
                    build_observations=observations,
                    build_actions=actions,
                    autosave_dir=Path(tmp),
                    data_device="cpu",
                    need_sample=True,
                )
            )
            replay.add_many(rows)
            subset = RawIndexBuffer(replay, name="test")
            subset.add_many([0, 1, 2])
            batch = subset.sample(4, "cpu", views=2)
            recent = subset.sample(32, "cpu", views=1, latest=1)
        self.assertEqual(batch["observation"].shape, (4, 2, 1))
        self.assertTrue(bool((recent["raw_id"] == 2).all()))
        self.assertTrue(bool((recent["observation"][:, 0, 0] == 2).all()))
        self.assertTrue(bool((batch["view_id"][:, 0] == 0).all()))
        self.assertTrue(bool((batch["next_view_id"][:, 0] == 0).all()))


if __name__ == "__main__":
    unittest.main()
