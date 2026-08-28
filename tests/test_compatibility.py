from __future__ import annotations

from dataclasses import replace

from ai_dj.matching import rank_next_tracks, score_track_pair
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


def test_close_tempo_and_compatible_keys_score_above_distant_tracks():
    source = _track("source", bpm=120.0, key="C major")
    compatible = _track("compatible", bpm=122.0, key="G major")
    distant = _track("distant", bpm=175.0, key="F# major", energy=0.2)

    compatible_result = score_track_pair(source, compatible)
    distant_result = score_track_pair(source, distant)

    assert compatible_result.overall_score > distant_result.overall_score
    assert _component(compatible_result, "tempo").score > 0.85
    assert _component(compatible_result, "harmony").score >= 0.85
    assert _component(distant_result, "tempo").score < 0.6
    assert _component(distant_result, "harmony").score < 0.5


def test_energy_and_vocal_overlap_components_are_explainable():
    vocals = VocalActivityEstimate((VocalActivity(0.0, 20.0, 1.0),), True, "fixture")
    source = _track("source", energy=0.8, vocal_activity=vocals)
    same_energy = _track("same", energy=0.79, vocal_activity=vocals)
    low_energy = _track("low", energy=0.15, vocal_activity=vocals)

    similar = score_track_pair(source, same_energy)
    mismatch = score_track_pair(source, low_energy)

    assert _component(similar, "energy").score > _component(mismatch, "energy").score
    assert _component(similar, "vocals").score == 0.0
    assert "Estimated vocal-overlap risk" in similar.weaknesses


def test_low_confidence_key_does_not_receive_effective_harmony_weight():
    source = _track("source", key="C major", key_confidence=0.95)
    low_confidence_candidate = _track("uncertain", key="F# major", key_confidence=0.05)

    result = score_track_pair(source, low_confidence_candidate)

    harmony = _component(result, "harmony")
    assert harmony.score < 0.5
    assert harmony.confidence == 0.05
    assert result.confidence < 1.0
    assert not any("Harmonic relation" in weakness for weakness in result.weaknesses)


def test_missing_features_are_neutral_and_confidence_weighted_not_fake_certainty():
    source = _track("source")
    missing = replace(
        _track("missing"),
        tempo=TempoEstimate(0.0, 0.0),
        key=KeyEstimate(None, 0.0),
        spectral=SpectralFeatures.zero(),
        structure=StructureAnalysis.empty(),
    )

    result = score_track_pair(source, missing)

    assert 0.0 <= result.overall_score <= 1.0
    assert result.confidence < 1.0
    assert _component(result, "tempo").confidence == 0.0
    assert _component(result, "harmony").confidence == 0.0
    assert _component(result, "timbre").confidence == 0.0


def test_ranking_orders_candidates_and_preserves_structured_explanations():
    source = _track("source", bpm=120.0, key="C major")
    strong = _track("strong", bpm=121.0, key="G major")
    weak = _track("weak", bpm=180.0, key="F# major", energy=0.1)

    ranked = rank_next_tracks(source, [weak, strong])

    assert [result.candidate_track_id for result in ranked] == ["strong", "weak"]
    assert len(ranked[0].components) == 7
    assert all(0.0 <= component.score <= 1.0 for component in ranked[0].components)
    assert all(0.0 <= component.confidence <= 1.0 for component in ranked[0].components)
    assert ranked[0].strengths


def _component(result, name: str):
    return next(component for component in result.components if component.name == name)


def _track(
    track_id: str,
    *,
    bpm: float = 120.0,
    key: str | None = "C major",
    key_confidence: float = 0.9,
    energy: float = 0.8,
    vocal_activity: VocalActivityEstimate | None = None,
) -> TrackAnalysis:
    duration = 20.0
    beat_times = tuple(index * 0.5 for index in range(40))
    downbeats = tuple(index * 2.0 for index in range(10))
    structure = StructureAnalysis(
        bars=tuple(Bar(start, start + 2.0, 0.9) for start in downbeats),
        phrases=(Phrase(0.0, 8.0, 0.9), Phrase(8.0, 16.0, 0.9), Phrase(16.0, 20.0, 0.5)),
        sections=(),
        vocal_activity=vocal_activity or VocalActivityEstimate.unavailable(),
    )
    return TrackAnalysis(
        track_id=track_id,
        source_path=f"{track_id}.wav",
        duration=duration,
        tempo=TempoEstimate(bpm, 0.9),
        beats=BeatEstimate(beat_times, 0.9),
        downbeats=DownbeatEstimate(downbeats, 0.9, 4),
        key=KeyEstimate(key, key_confidence),
        energy=EnergyEstimate(energy, (EnergyPoint(0.5, energy), EnergyPoint(19.5, energy))),
        spectral=SpectralFeatures(1000.0, 700.0, 2200.0, (10.0, 12.0), tuple(float(index) for index in range(13))),
        structure=structure,
    )
