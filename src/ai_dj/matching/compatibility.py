"""Transparent deterministic baseline for consecutive-track compatibility.

This module scores analyzed tracks only; it does not decode, alter, or render
audio. Component scores are in [0, 1]. Each score is weighted by the confidence
of the underlying analysis; unavailable information receives zero effective
weight rather than a fabricated positive or negative score.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ai_dj.matching.models import CompatibilityComponent, CompatibilityResult
from ai_dj.representation.track import TrackAnalysis

COMPONENT_WEIGHTS = {
    "tempo": 0.25,
    "harmony": 0.20,
    "rhythm": 0.15,
    "energy": 0.15,
    "structure": 0.10,
    "vocals": 0.05,
    "timbre": 0.10,
}
_NEUTRAL_SCORE = 0.5


def score_track_pair(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityResult:
    """Score a potential directed transition from ``source`` to ``candidate``.

    Overall score is the weighted mean of known component scores. Effective
    component weight equals its documented base weight times its confidence.
    Result confidence is the fraction of base-weighted analysis evidence present.
    """
    components = (
        _tempo_component(source, candidate),
        _harmony_component(source, candidate),
        _rhythm_component(source, candidate),
        _energy_component(source, candidate),
        _structure_component(source, candidate),
        _vocal_component(source, candidate),
        _timbre_component(source, candidate),
    )
    base_weight = sum(component.weight for component in components)
    effective_weight = sum(component.weight * component.confidence for component in components)
    overall = (
        sum(component.score * component.weight * component.confidence for component in components) / effective_weight
        if effective_weight > 0
        else _NEUTRAL_SCORE
    )
    strengths = tuple(component.reason for component in components if component.confidence >= 0.5 and component.score >= 0.75)
    weaknesses = tuple(component.reason for component in components if component.confidence >= 0.5 and component.score < 0.45)
    return CompatibilityResult(
        source_track_id=source.track_id,
        candidate_track_id=candidate.track_id,
        overall_score=round(float(np.clip(overall, 0.0, 1.0)), 3),
        confidence=round(float(np.clip(effective_weight / base_weight, 0.0, 1.0)), 3),
        components=components,
        strengths=strengths,
        weaknesses=weaknesses,
    )


def rank_next_tracks(current_track: TrackAnalysis, candidate_tracks: Iterable[TrackAnalysis]) -> list[CompatibilityResult]:
    """Score and return candidates in descending overall compatibility order."""
    results = [score_track_pair(current_track, candidate) for candidate in candidate_tracks]
    return sorted(results, key=lambda result: (-result.overall_score, -result.confidence, result.candidate_track_id))


def _tempo_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    first, second = source.tempo, candidate.tempo
    confidence = min(first.confidence, second.confidence)
    if first.bpm <= 0 or second.bpm <= 0:
        return _component("tempo", _NEUTRAL_SCORE, 0.0, "Tempo unavailable")
    ratio = max(first.bpm, second.bpm) / min(first.bpm, second.bpm)
    # Closest practical tempo ratio supports ordinary beat-matching and half/double time.
    adjustment = min(abs(np.log2(ratio / target)) for target in (1.0, 2.0, 0.5))
    score = float(np.exp(-5.0 * adjustment))
    return _component("tempo", score, confidence, f"Tempo adjustment ratio {ratio:.3f}")


def _harmony_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    confidence = min(source.key.confidence, candidate.key.confidence)
    if source.key.key is None or candidate.key.key is None:
        return _component("harmony", _NEUTRAL_SCORE, 0.0, "Key unavailable")
    source_tonic, source_mode = _parse_key(source.key.key)
    candidate_tonic, candidate_mode = _parse_key(candidate.key.key)
    if source_tonic is None or candidate_tonic is None:
        return _component("harmony", _NEUTRAL_SCORE, 0.0, "Key format unavailable")
    fifth_distance = min((source_tonic - candidate_tonic) % 12, (candidate_tonic - source_tonic) % 12)
    if source_tonic == candidate_tonic and source_mode == candidate_mode:
        score = 1.0
    elif source_tonic == candidate_tonic:
        score = 0.78
    elif source_mode != candidate_mode and fifth_distance in (3, 9):
        score = 0.85
    elif fifth_distance in (5, 7):
        score = 0.85 if source_mode == candidate_mode else 0.7
    elif fifth_distance in (3, 4, 8, 9):
        score = 0.65
    else:
        score = 0.25
    return _component("harmony", score, confidence, f"Harmonic relation: {source.key.key} → {candidate.key.key}")


def _rhythm_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    beat_confidence = min(source.beats.confidence, candidate.beats.confidence)
    downbeat_confidence = min(source.downbeats.confidence, candidate.downbeats.confidence)
    confidence = max(beat_confidence, downbeat_confidence)
    if confidence <= 0:
        return _component("rhythm", _NEUTRAL_SCORE, 0.0, "Beat grids unavailable")
    source_density = _event_density(source.beats.timestamps, source.duration)
    candidate_density = _event_density(candidate.beats.timestamps, candidate.duration)
    density_score = _ratio_score(source_density, candidate_density)
    source_meter, candidate_meter = source.downbeats.meter, candidate.downbeats.meter
    meter_score = 1.0 if source_meter and candidate_meter and source_meter == candidate_meter else 0.6
    score = 0.7 * density_score + 0.3 * meter_score
    return _component("rhythm", score, confidence, "Beat-grid density and inferred meter compatibility")


def _energy_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    confidence = 1.0 if source.energy.timeline and candidate.energy.timeline else 0.5
    global_score = float(np.clip(1.0 - abs(source.energy.global_level - candidate.energy.global_level) / 0.5, 0.0, 1.0))
    exit_level = _edge_energy(source, last=True)
    entry_level = _edge_energy(candidate, last=False)
    transition_score = float(np.clip(1.0 - abs(exit_level - entry_level) / 0.5, 0.0, 1.0))
    return _component("energy", 0.5 * global_score + 0.5 * transition_score, confidence, "Energy level and entry/exit continuity")


def _structure_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    source_structure, candidate_structure = source.structure, candidate.structure
    source_confidence = _structure_confidence(source)
    candidate_confidence = _structure_confidence(candidate)
    confidence = min(source_confidence, candidate_confidence)
    if confidence <= 0:
        return _component("structure", _NEUTRAL_SCORE, 0.0, "Phrase or bar candidates unavailable")
    phrase_score = 1.0 if source_structure.phrases and candidate_structure.phrases else 0.65
    meter_score = 1.0 if source.downbeats.meter == candidate.downbeats.meter else 0.65
    return _component("structure", 0.7 * phrase_score + 0.3 * meter_score, confidence, "Bar and phrase candidate compatibility")


def _vocal_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    source_vocals = source.structure.vocal_activity
    candidate_vocals = candidate.structure.vocal_activity
    if not source_vocals.available or not candidate_vocals.available:
        return _component("vocals", _NEUTRAL_SCORE, 0.0, "Vocal activity unavailable")
    source_probability = _mean_vocal_probability(source_vocals.segments, source.duration)
    candidate_probability = _mean_vocal_probability(candidate_vocals.segments, candidate.duration)
    overlap_risk = source_probability * candidate_probability
    return _component("vocals", 1.0 - overlap_risk, 1.0, "Estimated vocal-overlap risk")


def _timbre_component(source: TrackAnalysis, candidate: TrackAnalysis) -> CompatibilityComponent:
    first, second = source.spectral, candidate.spectral
    if not first.mfcc or not second.mfcc:
        return _component("timbre", _NEUTRAL_SCORE, 0.0, "Spectral summaries unavailable")
    vector_a = np.asarray((first.centroid_hz, first.bandwidth_hz, first.rolloff_hz, *first.contrast, *first.mfcc), dtype=float)
    vector_b = np.asarray((second.centroid_hz, second.bandwidth_hz, second.rolloff_hz, *second.contrast, *second.mfcc), dtype=float)
    length = min(len(vector_a), len(vector_b))
    similarity = _cosine_similarity(vector_a[:length], vector_b[:length])
    return _component("timbre", (similarity + 1.0) / 2.0, 1.0, "Spectral/timbral summary similarity")


def _component(name: str, score: float, confidence: float, reason: str) -> CompatibilityComponent:
    return CompatibilityComponent(
        name=name,
        score=round(float(np.clip(score, 0.0, 1.0)), 3),
        confidence=round(float(np.clip(confidence, 0.0, 1.0)), 3),
        weight=COMPONENT_WEIGHTS[name],
        reason=reason,
    )


def _parse_key(value: str) -> tuple[int | None, str | None]:
    pitches = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    parts = value.split(maxsplit=1)
    return (pitches.get(parts[0]), parts[1] if len(parts) == 2 else None)


def _event_density(timestamps: tuple[float, ...], duration: float) -> float:
    return len(timestamps) / duration if duration > 0 else 0.0


def _ratio_score(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.5
    return float(np.exp(-3.0 * abs(np.log(left / right))))


def _edge_energy(track: TrackAnalysis, *, last: bool) -> float:
    values = [point.value for point in track.energy.timeline]
    if not values:
        return track.energy.global_level
    return float(values[-1] if last else values[0])


def _structure_confidence(track: TrackAnalysis) -> float:
    values = [bar.confidence for bar in track.structure.bars]
    return float(np.mean(values)) if values else 0.0


def _mean_vocal_probability(segments, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return float(sum((segment.end - segment.start) * segment.probability for segment in segments) / duration)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
