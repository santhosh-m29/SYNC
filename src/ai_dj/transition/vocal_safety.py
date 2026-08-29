"""Deterministic, fail-closed vocal safety gate for transition plans.

This gate consumes *trusted* vocal timelines.  It deliberately does not try to
infer separate singers from a mixed rendered waveform: that cannot establish
whether two source tracks are simultaneously vocal.  If either timeline is
missing or too uncertain, a normal overlapping transition is not approved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_dj.representation.structure import VocalActivityEstimate
from ai_dj.representation.track import TrackAnalysis
from ai_dj.transition.models import TransitionPlan


@dataclass(frozen=True, slots=True)
class VocalSafetyConfig:
    """Conservative thresholds for validated vocal-activity timelines.

    These defaults are a safety policy, not a detector calibration.  A
    production detector must be calibrated against labelled validation audio
    before its timelines are marked trusted.
    """

    minimum_timeline_confidence: float = 0.60
    significant_probability: float = 0.60
    minimum_significant_seconds: float = 0.50
    maximum_collision_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class VocalSafetyResult:
    allowed: bool
    verified: bool
    collision_duration: float | None
    maximum_overlap_probability: float | None
    integrated_overlap: float | None
    reason: str


def tempo_stretch_rate(source_bpm: float, destination_bpm: float) -> float:
    """Match the renderer's pitch-preserving time-stretch convention."""
    if source_bpm <= 0.0 or destination_bpm <= 0.0:
        return 1.0
    ratio = source_bpm / destination_bpm
    relationship = min((0.5, 1.0, 2.0), key=lambda value: abs(ratio / value - 1.0))
    return ratio / relationship


def planned_beat_alignment(source: TrackAnalysis, destination: TrackAnalysis, plan: TransitionPlan, rate: float) -> float:
    """Return the renderer-equivalent bounded destination offset in seconds."""
    source_beat = _nearest(source.beats.timestamps, plan.source_exit)
    destination_beat = _nearest(destination.beats.timestamps, plan.destination_entry)
    if source_beat is None or destination_beat is None or rate <= 0.0:
        return 0.0
    return float(np.clip((source_beat - plan.source_exit) - (destination_beat - plan.destination_entry) / rate, -0.05, 0.05))


def assess_vocal_safety(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    plan: TransitionPlan,
    config: VocalSafetyConfig = VocalSafetyConfig(),
    *,
    destination_rate: float | None = None,
    destination_offset: float | None = None,
) -> VocalSafetyResult:
    """Fail closed unless the planned audible overlap is verified vocal-safe.

    A zero-duration handoff has no simultaneous source/destination audibility,
    so it is safe by construction even when a detector timeline is unavailable.
    """
    if plan.duration == 0.0:
        return VocalSafetyResult(True, True, 0.0, 0.0, 0.0, "Non-overlapping beat/phrase handoff")
    source_vocals = source.structure.vocal_activity
    destination_vocals = destination.structure.vocal_activity
    if not _trusted(source_vocals, config) or not _trusted(destination_vocals, config):
        return VocalSafetyResult(False, False, None, None, None, "Trusted vocal timelines are required for an overlapping transition")
    rate = destination_rate if destination_rate is not None else tempo_stretch_rate(source.tempo.bpm, destination.tempo.bpm)
    offset = destination_offset if destination_offset is not None else planned_beat_alignment(source, destination, plan, rate)
    source_segments = _mapped_segments(source_vocals, plan.source_exit, 1.0, 0.0, plan.duration, config)
    destination_segments = _mapped_segments(destination_vocals, plan.destination_entry, rate, offset, plan.duration, config)
    duration, maximum, integral = _overlap_metrics(source_segments, destination_segments, plan.duration)
    allowed = duration <= config.maximum_collision_seconds + 1e-9
    reason = "No simultaneous significant vocals" if allowed else f"Significant vocal collision for {duration:.3f}s (max probability product {maximum:.3f})"
    return VocalSafetyResult(allowed, True, round(duration, 6), round(maximum, 6), round(integral, 6), reason)


def _trusted(estimate: VocalActivityEstimate, config: VocalSafetyConfig) -> bool:
    return estimate.available and estimate.confidence >= config.minimum_timeline_confidence


def _mapped_segments(
    estimate: VocalActivityEstimate, entry: float, rate: float, offset: float, duration: float, config: VocalSafetyConfig
) -> tuple[tuple[float, float, float], ...]:
    if rate <= 0.0:
        return ()
    output: list[tuple[float, float, float]] = []
    for segment in estimate.segments:
        if segment.probability < config.significant_probability or segment.end - segment.start < config.minimum_significant_seconds:
            continue
        start = (segment.start - entry) / rate + offset
        end = (segment.end - entry) / rate + offset
        start, end = max(0.0, start), min(duration, end)
        if end - start >= config.minimum_significant_seconds:
            output.append((start, end, segment.probability))
    return tuple(output)


def _overlap_metrics(
    source: tuple[tuple[float, float, float], ...], destination: tuple[tuple[float, float, float], ...], duration: float
) -> tuple[float, float, float]:
    collision_duration = maximum = integral = 0.0
    for left_start, left_end, left_probability in source:
        for right_start, right_end, right_probability in destination:
            overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
            if overlap <= 0.0:
                continue
            product = left_probability * right_probability
            collision_duration += overlap
            maximum = max(maximum, product)
            integral += overlap * product
    return collision_duration, maximum, integral / max(duration, 1e-12)


def _nearest(values: tuple[float, ...], target: float) -> float | None:
    return min(values, key=lambda value: abs(value - target)) if values else None
