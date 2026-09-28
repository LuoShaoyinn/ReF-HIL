from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Callable, Iterator

import numpy as np
import torch

from shared.zmq import DictMessage


@dataclass(kw_only=True)
class DataBufferConfig:
    capacity:               int
    obs_dims:               int
    action_dims:            int
    info_features:          dict[str, tuple[int, ...]] = field(default_factory=dict)
    build_obs:              Callable[[DictMessage, DictMessage, bool], torch.Tensor] | None = None
    build_action:           Callable[[DictMessage, DictMessage, bool], torch.Tensor] | None = None
    build_observations:     Callable[..., torch.Tensor] | None = None
    build_actions:          Callable[..., torch.Tensor] | None = None
    autosave_dir:           str | Path = "outputs/default/replay"
    autosave_chunk_size:    int = 128

    data_device:            str = "cuda"
    need_sample:            bool = False


class DataBuffer:
    """
    Single buffer with integrated raw chunk autosave.

    - In-memory store keeps vectorized tensors for training/sampling.
    - Autosave writes raw transitions as pickled `list[dict]`.
    """

    def __init__(self, config: DataBufferConfig) -> None:
        self.config = config
        self.tot_transition = 0
        self.file_idx = 0

        self.data_obs:      torch.Tensor | None = None
        self.data_action:   torch.Tensor | None = None
        self.data_rew:      torch.Tensor | None = None
        self.data_done:     torch.Tensor | None = None
        self.data_info:     dict[str, torch.Tensor] = {}

        real_capacity = self.config.capacity
        if self.config.need_sample:
            self.data_obs = torch.zeros(
                (real_capacity, self.config.obs_dims),
                dtype=torch.float32,
                device=self.config.data_device,
            )
            self.data_action = torch.zeros(
                (real_capacity, self.config.action_dims),
                dtype=torch.float32,
                device=self.config.data_device,
            )
            self.data_rew  = torch.zeros(
                (real_capacity,),
                dtype=torch.float32,
                device=self.config.data_device
            )
            self.data_done = torch.zeros(
                (real_capacity,),
                dtype=torch.bool,
                device=self.config.data_device
            )
            self.data_info = {
                feat_key: torch.zeros(
                    (real_capacity, *feat_shape),
                    dtype=torch.float32,
                    device=self.config.data_device
                )
                for feat_key, feat_shape in self.config.info_features.items()
            }

        self.buffer: list[DictMessage] = []
        self.load()

    def digest_one_transition(self, transition: DictMessage) -> None:
        self.digest_transitions([transition])

    def digest_transitions(self, transitions: list[DictMessage]) -> list[int]:
        if not self.config.need_sample:
            return []
        raw_count = len(transitions)
        if raw_count == 0:
            return []
        assert self.tot_transition + raw_count <= self.config.capacity, \
            "buffer capacity exceeded"
        assert (
            self.data_obs is not None
            and self.data_action is not None
            and self.data_rew is not None
            and self.data_done is not None
        ), "buffer storage not initialized"

        assert self.config.build_obs is not None or self.config.build_observations is not None
        assert self.config.build_action is not None or self.config.build_actions is not None
        states = [transition["raw_obs"] for transition in transitions]
        infos = [transition["info"] for transition in transitions]
        raw_actions = [transition["raw_action"] for transition in transitions]
        first_idx = self.tot_transition
        idx = first_idx + torch.arange(raw_count, device=self.config.data_device)
        if self.config.build_observations is not None:
            obs = self.config.build_observations(states, infos, augment=False)
        else:
            obs = torch.stack(
                [self.config.build_obs(state, info, augment=False) for state, info in zip(states, infos, strict=True)],
                dim=0,
            )
        if self.config.build_actions is not None:
            action = self.config.build_actions(raw_actions, infos, augment=False)
        else:
            action = torch.stack(
                [self.config.build_action(raw_action, info, augment=False) for raw_action, info in zip(raw_actions, infos, strict=True)],
                dim=0,
            )

        self.data_obs[idx] = obs.to(self.config.data_device, non_blocking=True)
        self.data_action[idx] = action.to(self.config.data_device, non_blocking=True)
        self.data_rew[idx] = torch.as_tensor(
            [transition["reward"] for transition in transitions],
            dtype=torch.float32,
            device=self.config.data_device,
        )
        self.data_done[idx] = torch.as_tensor(
            [transition["done"] for transition in transitions],
            dtype=torch.bool,
            device=self.config.data_device,
        )
        for feat_key, feat_shape in self.config.info_features.items():
            feat_values = np.asarray(
                [transition["info"][feat_key] for transition in transitions], dtype=np.float32
            ).reshape(raw_count, *feat_shape)
            self.data_info[feat_key][idx] = torch.from_numpy(feat_values).to(
                self.config.data_device, non_blocking=True
            )
        self.tot_transition += raw_count
        return list(range(first_idx, first_idx + raw_count))

    def load(self) -> None:
        autosave_dir = Path(self.config.autosave_dir)
        autosave_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(autosave_dir.glob("*.pkl"))

        if self.config.need_sample:
            for path in paths:
                with path.open("rb") as f:
                    loaded = pickle.load(f)
                print(f"Loading {len(loaded)} transitions from {path}...")
                self.digest_transitions(loaded)

        self.file_idx = int(paths[-1].stem) + 1 if paths else 0
        self.buffer = []

    def save(self) -> None:
        if not self.buffer:
            return
        autosave_dir = Path(self.config.autosave_dir)
        autosave_dir.mkdir(parents=True, exist_ok=True)
        path = autosave_dir / f"{self.file_idx:04d}.pkl"
        print(f"Autosaving {len(self.buffer)} transitions to {path}...")
        with path.open("wb") as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.file_idx += 1
        self.buffer = []

    def add(self, transition: DictMessage) -> int:
        return self.add_many([transition])[0]

    def add_many(self, transitions: list[DictMessage]) -> list[int]:
        # Raw-only writers (the human recorder) do not build derived replay
        # rows, but ``add`` still has a stable integer return contract.
        transition_indices = (
            self.digest_transitions(transitions)
            if self.config.need_sample
            else [-1] * len(transitions)
        )
        self.buffer.extend(transitions)
        if len(self.buffer) >= self.config.autosave_chunk_size:
            self.save()
        return transition_indices

    def __del__(self) -> None:
        self.save()

    def __iter__(self) -> Iterator[tuple[int, DictMessage]]:
        assert self.config.need_sample, "__iter__() is unavailable when need_sample=False."
        for idx in range(self.tot_transition):
            yield idx, {
                "observation": self.data_obs[idx],
                "action": self.data_action[idx],
                "reward": self.data_rew[idx],
                "done": self.data_done[idx],
                "data_info": {
                    feat_key: feat_value[idx]
                    for feat_key, feat_value in self.data_info.items()
                },
            }

    def sample(self, batch_size: int, target_device: torch.device | str) -> DictMessage:
        """return a dict: ["observation", "action", "next_observation", "reward", "done", "data_info"]"""
        assert self.config.need_sample, "sample() is unavailable when need_sample=False."
        assert self.tot_transition >= 2, "sample() requires at least 2 transitions."

        positions = torch.randint(0, self.tot_transition - 1, (int(batch_size),), device=self.config.data_device)
        return self.sample_by_positions(positions, target_device)

    def sample_by_positions(self, positions: torch.Tensor, target_device: torch.device | str) -> DictMessage:
        """sample by given positions, used for prioritized replay or n-step return"""
        assert self.config.need_sample, "sample() is unavailable when need_sample=False."
        next_positions = torch.where(self.data_done[positions], positions, positions + 1)
        return {
            "observation":      self.data_obs[positions].to(target_device),
            "action":           self.data_action[positions].to(target_device),
            "next_observation": self.data_obs[next_positions].to(target_device),
            "reward":           self.data_rew[positions].to(target_device),
            "done":             self.data_done[positions].to(target_device),
            "data_info": {
                feat_key: feat_value[positions].to(target_device)
                for feat_key, feat_value in self.data_info.items()
            },
        }

    @property
    def raw_count(self) -> int:
        return self.tot_transition

    def raw_positions(
        self,
        raw_ids: torch.Tensor,
        view_ids: torch.Tensor | int = 0,
    ) -> torch.Tensor:
        view_tensor = torch.as_tensor(view_ids, device=raw_ids.device)
        return torch.broadcast_tensors(raw_ids, view_tensor)[0]

    def sample_raw_views(
        self,
        raw_ids: torch.Tensor,
        target_device: torch.device | str,
        *,
        views: int,
        include_clean: bool = True,
    ) -> DictMessage:
        """Sample grouped observation views without multiplying evidence count."""

        if views < 1:
            raise ValueError("views must be positive")
        raw_ids = raw_ids.to(self.config.data_device).long().reshape(-1)
        if bool((raw_ids < 0).any()) or bool((raw_ids >= self.raw_count).any()):
            raise IndexError("raw replay ID is out of range")
        batch = len(raw_ids)
        del include_clean
        view_ids = torch.zeros(
            (batch, views), dtype=torch.long, device=self.config.data_device
        )
        next_view_ids = torch.zeros_like(view_ids)
        positions = raw_ids[:, None].expand(-1, views)
        clean_positions = self.raw_positions(raw_ids)
        assert self.data_done is not None
        done = self.data_done[clean_positions]
        next_raw_ids = torch.where(done, raw_ids, raw_ids + 1)
        next_positions = next_raw_ids[:, None].expand(-1, views)
        next_clean_positions = self.raw_positions(next_raw_ids)
        return {
            "raw_id": raw_ids.to(target_device),
            "view_id": view_ids.to(target_device),
            "next_view_id": next_view_ids.to(target_device),
            "observation": self.data_obs[positions].to(target_device),
            "clean_observation": self.data_obs[clean_positions].to(target_device),
            "next_observation": self.data_obs[next_positions].to(target_device),
            "clean_next_observation": self.data_obs[next_clean_positions].to(target_device),
            "action": self.data_action[clean_positions].to(target_device),
            "reward": self.data_rew[clean_positions].to(target_device),
            "done": done.to(target_device),
            "data_info": {
                key: value[clean_positions].to(target_device)
                for key, value in self.data_info.items()
            },
        }
