"""Typed, inspectable representations for deterministic DJ-set search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ai_dj.transition.models import TransitionPlan

EnergyTrajectory = Literal["build", "maintain", "release", "peak"]


@dataclass(frozen=True, slots=True)
class SetPlanningConfig:
    target_track_count: int = 5
    beam_width: int = 8
    energy_trajectory: EnergyTrajectory = "maintain"
    artist_by_track_id: dict[str, str] | None = None
    recent_history_size: int = 3
    allow_track_repeats: bool = False


@dataclass(frozen=True, slots=True)
class SetStep:
    track_id: str
    transition: TransitionPlan
    transition_score: float
    energy_score: float
    variety_score: float
    artist_repeat_penalty: float
    similarity_penalty: float
    incremental_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "transition": self.transition.to_dict(),
            "objective": {
                "transition_score": self.transition_score,
                "energy_score": self.energy_score,
                "variety_score": self.variety_score,
                "artist_repeat_penalty": self.artist_repeat_penalty,
                "similarity_penalty": self.similarity_penalty,
                "incremental_score": self.incremental_score,
            },
        }


@dataclass(frozen=True, slots=True)
class SetPlan:
    track_ids: tuple[str, ...]
    steps: tuple[SetStep, ...]
    overall_score: float
    complete: bool
    energy_trajectory: EnergyTrajectory
    search_method: Literal["beam", "greedy"]
    notes: tuple[str, ...]

    @property
    def transitions(self) -> tuple[TransitionPlan, ...]:
        return tuple(step.transition for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracks": list(self.track_ids),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "overall_score": self.overall_score,
            "complete": self.complete,
            "energy_trajectory": self.energy_trajectory,
            "search_method": self.search_method,
            "steps": [step.to_dict() for step in self.steps],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PlanningComparison:
    greedy: SetPlan
    sequence_aware: SetPlan
