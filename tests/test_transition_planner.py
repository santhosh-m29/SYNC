from __future__ import annotations

from dataclasses import replace

from ai_dj.representation.structure import Bar, Phrase, StructureAnalysis, VocalActivity, VocalActivityEstimate
from ai_dj.representation.track import (
    BeatEstimate,
    DownbeatEstimate,
    EnergyEstimate,
    EnergyPoint,
    KeyEstimate,
    SpectralFeatures,
    TempoEstimate,
    TrackAnalysis,
)
from ai_dj.transition import find_best_transition, find_best_transitions


def test_transition_candidates_are_structural_valid_and_ranked():
    source = _track("source", duration=32.0)
    destination = _track("destination", duration=32.0)

    candidates = find_best_transitions(source, destination)

    assert candidates
    assert find_best_transition(source, destination) == candidates[0]
    assert candidates == sorted(candidates, key=lambda item: (-item.overall_score, -item.confidence, -item.duration, item.source_exit, item.destination_entry))
    for candidate in candidates:
        assert 0.0 <= candidate.source_exit < source.duration
        assert 0.0 <= candidate.destination_entry < destination.duration
        assert candidate.source_exit + candidate.duration <= source.duration
        assert candidate.destination_entry + candidate.duration <= destination.duration
        assert candidate.strategy in {"phrase_crossfade", "instrumental_entry", "outro_intro", "energy_drop", "energy_rise"}
        assert len(candidate.components) == 7
        assert all(0.0 <= component.score <= 1.0 for component in candidate.components)
        assert all(0.0 <= component.confidence <= 1.0 for component in candidate.components)
    assert candidates[0].strategy == "phrase_crossfade"
    assert candidates[0].to_dict()["source"] == "source"


def test_planner_uses_vocal_activity_when_available_and_neutralizes_when_missing():
    full_vocals = VocalActivityEstimate((VocalActivity(0.0, 32.0, 1.0),), True, "fixture")
    source = _track("source", duration=32.0, vocal_activity=full_vocals)
    destination = _track("destination", duration=32.0, vocal_activity=full_vocals)

    plan = find_best_transition(source, destination)

    assert plan is not None
    vocal = _component(plan, "vocals")
    assert vocal.score == 0.0
    assert vocal.confidence == 1.0

    unavailable = find_best_transition(_track("a", duration=32.0), _track("b", duration=32.0))
    assert unavailable is not None
    vocal = _component(unavailable, "vocals")
    assert vocal.score == 0.5
    assert vocal.confidence == 0.0


def test_short_or_unstructured_tracks_produce_no_fabricated_transition_points():
    structured = _track("structured", duration=32.0)
    short = replace(_track("short", duration=3.0), structure=StructureAnalysis.empty(), downbeats=DownbeatEstimate((), 0.0, None))

    assert find_best_transitions(short, structured) == []
    assert find_best_transition(structured, short) is None


def test_low_confidence_analysis_remains_bounded_and_reduces_plan_confidence():
    source = _track("source", duration=32.0, confidence=0.9)
    uncertain = _track("uncertain", duration=32.0, confidence=0.1)

    strong_plan = find_best_transition(source, _track("strong", duration=32.0, confidence=0.9))
    uncertain_plan = find_best_transition(source, uncertain)

    assert strong_plan is not None and uncertain_plan is not None
    assert 0.0 <= uncertain_plan.overall_score <= 1.0
    assert uncertain_plan.confidence < strong_plan.confidence


def _component(plan, name: str):
    return next(component for component in plan.components if component.name == name)


def _track(
    track_id: str,
    *,
    duration: float,
    confidence: float = 0.9,
    vocal_activity: VocalActivityEstimate | None = None,
) -> TrackAnalysis:
    downbeats = tuple(float(value) for value in range(0, int(duration), 2))
    bars = tuple(Bar(start, min(start + 2.0, duration), confidence) for start in downbeats if start + 2.0 <= duration)
    phrases = tuple(Phrase(start, min(start + 8.0, duration), confidence) for start in range(0, int(duration), 8))
    beat_times = tuple(float(value) for value in range(0, int(duration * 2)) )
    energy_points = (EnergyPoint(0.5, 0.7), EnergyPoint(max(duration - 0.5, 0.5), 0.7))
    return TrackAnalysis(
        track_id=track_id,
        source_path=f"{track_id}.wav",
        duration=duration,
        tempo=TempoEstimate(120.0, confidence),
        beats=BeatEstimate(beat_times, confidence),
        downbeats=DownbeatEstimate(downbeats, confidence, 4),
        key=KeyEstimate("C major", confidence),
        energy=EnergyEstimate(0.7, energy_points),
        spectral=SpectralFeatures(1000.0, 700.0, 2200.0, (10.0,), tuple(float(value) for value in range(13))),
        structure=StructureAnalysis(bars=bars, phrases=phrases, sections=(), vocal_activity=vocal_activity or VocalActivityEstimate.unavailable()),
    )
