"""Typed, explainable specifications for planned—not rendered—transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TransitionComponent:
    name: str
    score: float
    confidence: float
    weight: float
    reason: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "score": self.score,
            "confidence": self.confidence,
            "weight": self.weight,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    source_track_id: str
    destination_track_id: str
    source_exit: float
    destination_entry: float
    duration: float
    strategy: str
    overall_score: float
    confidence: float
    components: tuple[TransitionComponent, ...]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_track_id,
            "destination": self.destination_track_id,
            "source_exit": self.source_exit,
            "destination_entry": self.destination_entry,
            "duration": self.duration,
            "strategy": self.strategy,
            "score": self.overall_score,
            "confidence": self.confidence,
            "components": [component.to_dict() for component in self.components],
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
        }
