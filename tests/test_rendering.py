from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from ai_dj.rendering import RenderConfig, RenderError, render_transition
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
