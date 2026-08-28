from __future__ import annotations

import importlib
import json
from dataclasses import replace

import numpy as np
import soundfile as sf

from ai_dj.pipeline.analyze import LibraryAnalysisResult
from ai_dj.pipeline.generate import GenerationConfig, generate_dj_set, retrieve_candidates
from ai_dj.rendering import RenderConfig
from tests.test_transition_planner import _track


def test_end_to_end_generation_uses_cached_analyses_and_writes_reports(tmp_path, monkeypatch):
    tracks = _tracks_with_audio(tmp_path)
    result = LibraryAnalysisResult(analyses=tracks, failures=[], cached=3, skipped=0)
    module = importlib.import_module("ai_dj.pipeline.generate")
    monkeypatch.setattr(module, "analyze_library", lambda *_: result)
    monkeypatch.setattr(module, "RenderConfig", lambda **_: RenderConfig(sample_rate=8_000, source_pre_roll_seconds=1.0, destination_post_roll_seconds=1.0))

    generated = generate_dj_set(
        tmp_path,
        tmp_path / "output" / "set.wav",
        GenerationConfig(target_track_count=3, candidate_pool_size=2, beam_width=2, energy_trajectory="build", sample_rate=8_000),
    )
    samples, sample_rate = sf.read(generated.output_path, dtype="float32")
    report = json.loads(generated.report_path.read_text(encoding="utf-8"))

    assert generated.output_path.is_file()
    assert generated.report_path.is_file()
    assert generated.output_path.with_suffix(".report.txt").is_file()
    assert "Transitions:" in generated.output_path.with_suffix(".report.txt").read_text(encoding="utf-8")
    assert len(generated.set_plan.track_ids) == 3
    assert len(set(generated.set_plan.track_ids)) == 3
    assert generated.metrics["transitions_generated"] == 2
    assert generated.metrics["technical_failures"] == 0
    assert generated.fallbacks == ("deterministic_transition_and_set_scoring",)
    assert sample_rate == 8_000 and np.isfinite(samples).all()
    assert report["metrics"]["quality"]["clipping_samples"] == 0
    assert report["metrics"]["quality"]["join_quality"]["count"] == 1
    assert report["metrics"]["quality"]["transition_beat_alignment_max_seconds"] <= 0.05
    assert samples.size == sum(int(round(item.duration * sample_rate)) for item in generated.render_results) - int(0.02 * sample_rate)
    assert report["analysis"]["cached"] == 3


def test_candidate_retrieval_is_bounded_and_deterministic(tmp_path):
    tracks = _tracks_with_audio(tmp_path)
    assert retrieve_candidates(tracks[0], list(reversed(tracks)), 1) == retrieve_candidates(tracks[0], tracks, 1)
    assert len(retrieve_candidates(tracks[0], tracks, 1)) == 1


def test_equal_power_set_join_reduces_an_abrupt_boundary(tmp_path):
    from ai_dj.pipeline.generate import _assemble_set
    from ai_dj.rendering import RenderResult

    paths = [tmp_path / "positive.wav", tmp_path / "negative.wav"]
    for path, level in zip(paths, (1.0, -1.0), strict=True):
        sf.write(path, np.full(100, level, dtype=np.float32), 1_000, subtype="FLOAT")
    renders = tuple(
        RenderResult(path, 1_000, 0.1, 0.0, 0.0, "phrase_crossfade", 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0)
        for path in paths
    )
    metrics = _assemble_set(renders, tmp_path / "joined.wav", 1_000, 0.02)
    assert metrics["max_boundary_jump"] < 0.2


def _tracks_with_audio(tmp_path):
    tracks = []
    for index, level in enumerate((0.35, 0.55, 0.75)):
        path = tmp_path / f"track-{index}.wav"
        _write_tone(path, level)
        base = _track(f"track-{index}", duration=32.0)
        tracks.append(replace(base, source_path=str(path), energy=replace(base.energy, global_level=level)))
    return tracks


def _write_tone(path, amplitude):
    sample_rate = 8_000
    time = np.arange(sample_rate * 32, dtype=np.float32) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * 220 * time)
    sf.write(path, samples, sample_rate, subtype="FLOAT")
