from __future__ import annotations

from types import SimpleNamespace

import torch

from .replay import DataBuffer


class RawIndexBuffer:
    """Index-only view whose entries are physical raw transition IDs."""

    def __init__(self, main_buffer: DataBuffer, *, name: str) -> None:
        self.main_buffer = main_buffer
        self.name = name
        self.idx = torch.zeros(
            (main_buffer.config.capacity,),
            dtype=torch.long,
            device=main_buffer.config.data_device,
        )
        self.count = 0

    def clear(self) -> None:
        self.count = 0

    def add(self, raw_id: int) -> None:
        if self.count >= len(self.idx):
            raise RuntimeError(f"{self.name} capacity exceeded")
        self.idx[self.count] = int(raw_id)
        self.count += 1

    def add_many(self, raw_ids: list[int] | torch.Tensor) -> None:
        values = torch.as_tensor(
            raw_ids,
            dtype=torch.long,
            device=self.idx.device,
        ).reshape(-1)
        if self.count + len(values) > len(self.idx):
            raise RuntimeError(f"{self.name} capacity exceeded")
        self.idx[self.count : self.count + len(values)] = values
        self.count += len(values)

    @property
    def raw_ids(self) -> torch.Tensor:
        return self.idx[: self.count]

    def sample_raw_ids(
        self, count: int, *, latest: int | None = None
    ) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError(f"cannot sample empty {self.name}")
        if latest is not None and latest < 1:
            raise ValueError("latest must be positive")
        first = 0 if latest is None else max(0, self.count - int(latest))
        selected = torch.randint(
            first,
            self.count,
            (int(count),),
            device=self.idx.device,
        )
        return self.idx[selected]

    def sample(
        self,
        count: int,
        target_device: torch.device | str,
        *,
        views: int,
        include_clean: bool = True,
        latest: int | None = None,
    ):
        return self.main_buffer.sample_raw_views(
            self.sample_raw_ids(count, latest=latest),
            target_device,
            views=views,
            include_clean=include_clean,
        )


class RawIndexDataView(RawIndexBuffer):
    """Compatibility view exposing clean rows of an index-only subset."""

    def __init__(self, main_buffer: DataBuffer, *, name: str) -> None:
        super().__init__(main_buffer, name=name)
        self.config = SimpleNamespace(
            capacity=main_buffer.config.capacity,
            data_device=main_buffer.config.data_device,
        )

    @property
    def tot_transition(self) -> int:
        return self.count

    @property
    def _clean_positions(self) -> torch.Tensor:
        return self.main_buffer.raw_positions(self.raw_ids)

    @property
    def data_done(self) -> torch.Tensor:
        return self.main_buffer.data_done[self._clean_positions]

    @property
    def data_rew(self) -> torch.Tensor:
        return self.main_buffer.data_rew[self._clean_positions]

    @property
    def data_action(self) -> torch.Tensor:
        return self.main_buffer.data_action[self._clean_positions]

    @property
    def data_info(self) -> dict[str, torch.Tensor]:
        positions = self._clean_positions
        return {
            key: value[positions]
            for key, value in self.main_buffer.data_info.items()
        }

    def sample_by_positions(
        self,
        positions: torch.Tensor,
        target_device: torch.device | str,
    ):
        raw_ids = self.raw_ids[positions.to(self.idx.device)]
        grouped = self.main_buffer.sample_raw_views(
            raw_ids,
            target_device,
            views=1,
            include_clean=True,
        )
        return {
            "observation": grouped["observation"][:, 0],
            "next_observation": grouped["next_observation"][:, 0],
            "action": grouped["action"],
            "reward": grouped["reward"],
            "done": grouped["done"],
            "data_info": grouped["data_info"],
        }

    def save(self) -> None:
        pass

class SubsetDataBuffer(DataBuffer):
    """A view of a subset of the main buffer, for multi-agent scenarios."""
    def __init__(self, main_buffer: DataBuffer) -> None:
        self.main_buffer = main_buffer
        self.config = main_buffer.config
        self.idx = torch.zeros((self.config.capacity,),
                               dtype=torch.long, device=self.config.data_device)
        self.tot_transition = 0
        assert main_buffer.config.need_sample, \
            "SubsetDataBuffer requires the main buffer to have need_sample=True."

    def __del__(self) -> None:
        self.save()

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass

    def add(self, transition_idx: int) -> int:
        assert self.tot_transition < self.config.capacity, "SubsetDataBuffer capacity exceeded."
        self.idx[self.tot_transition] = transition_idx
        self.tot_transition += 1
        return self.tot_transition - 1

    def sample(self, batch_size, target_device):
        assert self.tot_transition >= 2, "sample() requires at least 2 transitions."
        assert batch_size <= self.tot_transition, (
            "batch_size exceeds the number of transitions in SubsetDataBuffer."
        )
        subset_idx = torch.randint(0, self.tot_transition - 1, (batch_size,), device=self.config.data_device)
        return self.main_buffer.sample_by_positions(self.idx[subset_idx], target_device)
