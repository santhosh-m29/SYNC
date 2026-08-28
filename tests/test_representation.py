from __future__ import annotations

from ai_dj.representation.json_io import read_analysis, write_analysis
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
from ai_dj.representation.structure import Bar, Phrase, Section, StructureAnalysis, VocalActivityEstimate


def test_track_analysis_round_trips_through_dict_and_json(tmp_path):
    analysis = TrackAnalysis(
        track_id="track-id",
        source_path="example.wav",
        duration=12.5,
        tempo=TempoEstimate(128.0, 0.9, (64.0, 256.0)),
        beats=BeatEstimate((0.0, 0.5, 1.0), 0.8),
        downbeats=DownbeatEstimate((0.0,), 0.7, 4),
        key=KeyEstimate("A minor", 0.6),
        energy=EnergyEstimate(0.8, (EnergyPoint(0.5, 0.7), EnergyPoint(1.5, 0.9))),
        spectral=SpectralFeatures(1200.0, 900.0, 2500.0, (10.0, 12.0), tuple(range(13))),
        structure=StructureAnalysis(
            bars=(Bar(0.0, 2.0, 0.8),),
            phrases=(Phrase(0.0, 2.0, 0.7),),
            sections=(Section(0.0, 2.0, "other", "section_1", 0.6),),
            vocal_activity=VocalActivityEstimate.unavailable(),
        ),
    )

    assert TrackAnalysis.from_dict(analysis.to_dict()) == analysis
    destination = tmp_path / "analysis.json"
    write_analysis(destination, analysis)
    assert read_analysis(destination) == analysis
