from __future__ import annotations

import numpy as np
import pytest

from ai_dj.analysis.beats import estimate_beats
from ai_dj.analysis.downbeats import estimate_downbeats
from ai_dj.analysis.energy import estimate_energy
from ai_dj.analysis.key import estimate_key
from ai_dj.analysis.spectral import estimate_spectral_features
from ai_dj.analysis.tempo import estimate_tempo
from ai_dj.ingestion.loader import load_audio
from ai_dj.pipeline.analyze import analyze_track
from ai_dj.representation.track import ANALYSIS_VERSION
from tests.conftest import write_click_track


def test_tempo_result_for_click_track_is_structured_and_reasonable(tmp_path):
    audio = load_audio(write_click_track(tmp_path / "clicks.wav", bpm=120.0, duration=12.0))

    result = estimate_tempo(audio.samples, audio.sample_rate)

    assert result.bpm == pytest.approx(120.0, abs=6.0)
    assert 0.0 <= result.confidence <= 1.0
    assert all(candidate > 0 for candidate in result.octave_alternatives)


def test_tempo_and_beats_handle_silence_without_false_certainty():
    silence = np.zeros(22_050, dtype=np.float32)

    tempo = estimate_tempo(silence, 22_050)
    beats = estimate_beats(silence, 22_050)
    downbeats = estimate_downbeats(silence, 22_050, beats)
    key = estimate_key(silence, 22_050)
    energy = estimate_energy(silence, 22_050)
    spectral = estimate_spectral_features(silence, 22_050)

    assert tempo.bpm == 0.0
    assert tempo.confidence == 0.0
    assert beats.timestamps == ()
    assert beats.confidence == 0.0
    assert downbeats.timestamps == ()
    assert downbeats.confidence == 0.0
    assert key.key is None
    assert key.confidence == 0.0
    assert energy.global_level == 0.0
    assert spectral.centroid_hz == 0.0


def test_beat_timestamps_are_monotonic_and_within_duration(tmp_path):
    audio = load_audio(write_click_track(tmp_path / "clicks.wav", bpm=128.0, duration=12.0))

    result = estimate_beats(audio.samples, audio.sample_rate)

    assert len(result.timestamps) >= 8
    assert list(result.timestamps) == sorted(result.timestamps)
    assert all(0.0 <= timestamp <= audio.duration for timestamp in result.timestamps)
    assert 0.0 <= result.confidence <= 1.0


def test_downbeat_estimation_uses_accent_pattern_without_assuming_one_fixed_meter():
    sample_rate = 22_050
    samples = _accented_click_signal(sample_rate=sample_rate, meter=4)
    beats = estimate_beats(samples, sample_rate)

    result = estimate_downbeats(samples, sample_rate, beats)

    assert result.meter == 4
    assert len(result.timestamps) >= 4
    assert list(result.timestamps) == sorted(result.timestamps)
    assert 0.0 <= result.confidence <= 1.0


def test_key_energy_and_spectral_results_are_structured_for_deterministic_signals():
    sample_rate = 22_050
    seconds = 4.0
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    # A C-major scale is less key-ambiguous than a single C-major triad.
    scale = [261.626, 293.665, 329.628, 349.228, 391.995, 440.0, 493.883, 523.251]
    key_signal = np.concatenate(
        [0.35 * np.sin(2 * np.pi * frequency * np.arange(sample_rate) / sample_rate) for frequency in scale]
    )
    chord = sum(0.2 * np.sin(2 * np.pi * frequency * time) for frequency in (261.626, 329.628, 391.995))
    bright_tone = chord + 0.1 * np.sin(2 * np.pi * 880.0 * time)

    key = estimate_key(key_signal.astype(np.float32), sample_rate)
    quiet_energy = estimate_energy((bright_tone * 0.05).astype(np.float32), sample_rate)
    loud_energy = estimate_energy(bright_tone.astype(np.float32), sample_rate)
    spectral = estimate_spectral_features(bright_tone.astype(np.float32), sample_rate)

    assert key.key == "C major"
    assert 0.0 <= key.confidence <= 1.0
    assert loud_energy.global_level > quiet_energy.global_level > 0.0
    assert len(loud_energy.timeline) == 4
    assert all(0.0 <= point.value <= 1.0 for point in loud_energy.timeline)
    assert spectral.centroid_hz > 200.0
    assert spectral.bandwidth_hz > 0.0
    assert spectral.rolloff_hz > 0.0
    assert len(spectral.contrast) > 0
    assert len(spectral.mfcc) == 13


def test_track_pipeline_populates_all_phase_one_features(tmp_path):
    analysis = analyze_track(write_click_track(tmp_path / "clicks.wav", bpm=128.0, duration=8.0))

    assert analysis.analysis_version == ANALYSIS_VERSION
    assert 0.0 <= analysis.downbeats.confidence <= 1.0
    assert 0.0 <= analysis.key.confidence <= 1.0
    assert 0.0 <= analysis.energy.global_level <= 1.0
    assert analysis.energy.timeline
    assert len(analysis.spectral.mfcc) == 13


def _accented_click_signal(*, sample_rate: int, meter: int) -> np.ndarray:
    duration = 12.0
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    click_length = int(0.02 * sample_rate)
    beat_samples = int(0.5 * sample_rate)
    for beat, start in enumerate(range(0, len(samples), beat_samples)):
        end = min(start + click_length, len(samples))
        amplitude = 0.9 if beat % meter == 0 else 0.2
        samples[start:end] = amplitude * np.hanning((end - start) * 2)[: end - start]
    return samples
