from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from ai_dj.rendering import RenderConfig, RenderError, StemPaths, render_transition
from ai_dj.representation.track import TempoEstimate
from ai_dj.transition import find_best_transition
from tests.test_transition_planner import _track


@pytest.mark.parametrize("strategy", ["phrase_crossfade", "instrumental_entry", "outro_intro"])
def test_renderer_writes_safe_lossless_transition_preview(tmp_path, strategy):
    source, destination, plan = _render_fixture(tmp_path, destination_bpm=126.0)
    plan = replace(plan, strategy=strategy)
    output = tmp_path / f"{strategy}.wav"
    config = RenderConfig(sample_rate=8_000, source_pre_roll_seconds=2.0, destination_post_roll_seconds=3.0)

    result = render_transition(source, destination, plan, output, config)
    samples, sample_rate = sf.read(output, dtype="float32")

    assert output.is_file()
    assert sample_rate == 8_000 == result.sample_rate
    assert samples.size == int(round((2.0 + plan.duration + 3.0) * sample_rate))
    assert result.duration == pytest.approx(samples.size / sample_rate, abs=1e-6)
    assert result.transition_start == 2.0
    assert result.time_stretch_rate == pytest.approx(120.0 / 126.0, abs=1e-6)
    assert np.isfinite(samples).all()
    assert result.peak <= config.peak_ceiling
    assert result.clipping_samples == 0
    assert result.rms > 0.01


def test_renderer_rejects_invalid_or_extreme_tempo_plans(tmp_path):
    source, destination, plan = _render_fixture(tmp_path)
    extreme = replace(destination, tempo=TempoEstimate(170.0, 0.9))
    with pytest.raises(RenderError, match="ineligible"):
        render_transition(source, extreme, plan, tmp_path / "bad.wav", RenderConfig(sample_rate=8_000))
    with pytest.raises(RenderError, match="Unsupported"):
        render_transition(source, destination, replace(plan, strategy="energy_rise"), tmp_path / "bad-strategy.wav", RenderConfig(sample_rate=8_000))


def test_renderer_peak_protection_and_plan_identifiers(tmp_path):
    source, destination, plan = _render_fixture(tmp_path, amplitude=1.0)
    result = render_transition(
        source, destination, plan, tmp_path / "protected.wav", RenderConfig(sample_rate=8_000, peak_ceiling=0.7)
    )
    assert result.peak <= 0.7
    assert result.peak_protection_db < 0.0
    wrong_plan = replace(plan, source_track_id="wrong")
    with pytest.raises(RenderError, match="identifiers"):
        render_transition(source, destination, wrong_plan, tmp_path / "wrong.wav", RenderConfig(sample_rate=8_000))


def test_stem_vocal_handoff_delays_destination_vocals_and_releases_source_vocals(tmp_path):
    source, destination, plan = _render_fixture(tmp_path)
    source_stems = _write_stems(tmp_path, "source", instrumental=0.1, vocals=0.3)
    destination_stems = _write_stems(tmp_path, "destination", instrumental=0.1, vocals=0.4)
    config = RenderConfig(sample_rate=8_000, source_pre_roll_seconds=2.0, destination_post_roll_seconds=2.0)

    result = render_transition(
        source, destination, plan, tmp_path / "handoff.wav", config,
        source_stems=source_stems, destination_stems=destination_stems,
    )
    samples, sample_rate = sf.read(result.output_path, dtype="float32")
    release = int(plan.duration * sample_rate * config.source_vocal_release_fraction)
    entry = int(plan.duration * sample_rate * config.destination_vocal_entry_fraction)
    transition_start = int(result.transition_start * sample_rate)

    assert result.vocal_handoff_applied
    assert result.destination_vocal_entry_seconds == pytest.approx(entry / sample_rate, abs=1e-6)
    # The source vocal has reached zero before the destination vocal begins;
    # the instrumental mix continues through the entire overlap.
    assert release <= entry
    assert samples[transition_start + entry - 1] == pytest.approx(0.1 * np.cos(np.pi * (entry - 1) / (2 * (len(samples) - transition_start - 2))) + 0.1 * np.sin(np.pi * (entry - 1) / (2 * (len(samples) - transition_start - 2))), abs=0.2)
    assert np.isfinite(samples).all()


def test_stem_handoff_requires_stems_for_both_tracks(tmp_path):
    source, destination, plan = _render_fixture(tmp_path)
    stems = _write_stems(tmp_path, "source", instrumental=0.1, vocals=0.3)
    with pytest.raises(RenderError, match="both source and destination"):
        render_transition(source, destination, plan, tmp_path / "bad-stems.wav", RenderConfig(sample_rate=8_000), source_stems=stems)


def _render_fixture(tmp_path, *, destination_bpm=120.0, amplitude=0.5):
    sample_rate = 8_000
    duration = 32.0
    source_path = tmp_path / "source.wav"
    destination_path = tmp_path / "destination.wav"
    _write_tone(source_path, sample_rate, duration, 220.0, amplitude)
    _write_tone(destination_path, sample_rate, duration, 330.0, amplitude * 0.5)
    source = replace(_track("source", duration=duration), source_path=str(source_path))
    destination = replace(
        _track("destination", duration=duration), source_path=str(destination_path), tempo=TempoEstimate(destination_bpm, 0.9)
    )
    plan = find_best_transition(source, destination)
    assert plan is not None
    return source, destination, plan


def _write_tone(path, sample_rate, duration, frequency, amplitude):
    times = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    clicks = (np.mod(times, 0.5) < 0.01).astype(np.float32) * 0.15
    samples = np.clip(amplitude * np.sin(2.0 * np.pi * frequency * times) + clicks, -1.0, 1.0)
    sf.write(path, samples, sample_rate, subtype="FLOAT")


def _write_stems(tmp_path, name, *, instrumental, vocals):
    sample_rate = 8_000
    count = sample_rate * 32
    instrumental_path = tmp_path / f"{name}-instrumental.wav"
    vocal_path = tmp_path / f"{name}-vocals.wav"
    sf.write(instrumental_path, np.full(count, instrumental, dtype=np.float32), sample_rate, subtype="FLOAT")
    sf.write(vocal_path, np.full(count, vocals, dtype=np.float32), sample_rate, subtype="FLOAT")
    return StemPaths(instrumental_path, vocal_path)
