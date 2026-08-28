"""Typed outputs for deterministic track-pair compatibility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompatibilityComponent:
    """One [0, 1] component score and the confidence used to weight it."""

    name: str
    score: float
    confidence: float
    weight: float
    reason: str


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Complete, explainable ranking result for a directed track pair."""

    source_track_id: str
    candidate_track_id: str
    overall_score: float
    confidence: float
    components: tuple[CompatibilityComponent, ...]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
