from __future__ import annotations

import numpy as np

from ai_dj.analysis.energy import estimate_energy
from ai_dj.analysis.structure import analyze_structure, infer_bars, infer_phrases
from ai_dj.analysis.vocals import estimate_vocal_activity
from ai_dj.representation.track import DownbeatEstimate


def test_structure_analysis_finds_bar_aligned_changes_and_repetition():
    sample_rate = 22_050
    samples = _repeating_sections_signal(sample_rate)
    duration = len(samples) / sample_rate
    downbeats = DownbeatEstimate(tuple(float(value) for value in range(0, 24, 2)), 0.9, 4)

    result = analyze_structure(samples, sample_rate, duration, downbeats, estimate_energy(samples, sample_rate))

    assert len(result.bars) == 12
    assert len(result.phrases) == 3
    assert [(section.start, section.end) for section in result.sections] == [(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)]
    assert [section.label for section in result.sections] == ["other", "other", "other"]
    assert [section.repetition_id for section in result.sections] == ["section_1", "section_2", "section_1"]
    _assert_ordered_ranges(result.bars, duration)
    _assert_ordered_ranges(result.phrases, duration)
    _assert_ordered_ranges(result.sections, duration)
    assert all(0.0 <= item.confidence <= 1.0 for item in (*result.bars, *result.phrases, *result.sections))


def test_structure_handles_missing_downbeats_without_inventing_sections():
    samples = np.zeros(22_050, dtype=np.float32)
    result = analyze_structure(
        samples,
        22_050,
        1.0,
        DownbeatEstimate((), 0.0, None),
        estimate_energy(samples, 22_050),
    )

    assert result.bars == ()
    assert result.phrases == ()
    assert result.sections == ()


def test_phrase_grouping_is_deterministic_and_uses_bar_candidates():
    downbeats = DownbeatEstimate(tuple(float(value) for value in range(0, 10, 2)), 0.8, 3)
    bars = infer_bars(10.0, downbeats)
    phrases = infer_phrases(bars)

    assert [(phrase.start, phrase.end) for phrase in phrases] == [(0.0, 8.0), (8.0, 10.0)]
    assert phrases[0].confidence == 0.8
    assert 0.0 < phrases[1].confidence < phrases[0].confidence


def test_vocal_activity_is_explicitly_unavailable_without_a_validated_local_model():
    estimate = estimate_vocal_activity(np.zeros(22_050, dtype=np.float32), 22_050)

    assert estimate.available is False
    assert estimate.method == "unavailable"
    assert estimate.segments == ()


def _repeating_sections_signal(sample_rate: int) -> np.ndarray:
    time = np.arange(sample_rate * 8) / sample_rate
    section_a = 0.25 * np.sin(2 * np.pi * 220.0 * time)
    section_b = 0.25 * np.sin(2 * np.pi * 880.0 * time)
    return np.concatenate((section_a, section_b, section_a)).astype(np.float32)


def _assert_ordered_ranges(items, duration: float) -> None:
    previous_end = 0.0
    for item in items:
        assert 0.0 <= item.start < item.end <= duration
        assert item.start >= previous_end
        previous_end = item.end
