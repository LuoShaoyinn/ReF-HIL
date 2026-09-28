"""Gear-insertion learner configuration."""

from __future__ import annotations

from dataclasses import dataclass

from learner.training import LimitActionLearnerConfig


@dataclass(kw_only=True)
class LearnerConfig(LimitActionLearnerConfig):
    pass
