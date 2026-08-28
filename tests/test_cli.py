from __future__ import annotations

import pytest

from ai_dj.cli.main import main
from ai_dj.representation.json_io import read_analysis
from tests.conftest import write_click_track


def test_cli_help_exits_successfully():
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0


def test_cli_analyzes_audio_writes_json_and_reuses_cache(tmp_path):
    music = tmp_path / "music"
    write_click_track(music / "song.wav", bpm=120.0)
    cache = tmp_path / "cache"
    output = tmp_path / "output"

    assert main(["analyze", str(music), "--cache-dir", str(cache), "--output-dir", str(output)]) == 0
    generated = list(output.glob("*.json"))
    assert len(generated) == 1
    assert read_analysis(generated[0]).tempo.bpm > 0

    assert main(["analyze", str(music), "--cache-dir", str(cache), "--output-dir", str(output)]) == 0


def test_cli_continues_after_a_bad_file(tmp_path):
    music = tmp_path / "music"
    write_click_track(music / "valid.wav")
    (music / "broken.wav").write_bytes(b"invalid audio")

    exit_code = main(["analyze", str(music), "--cache-dir", str(tmp_path / "cache")])

    assert exit_code == 1
    assert list((music / ".ai_dj_analysis").glob("*.json"))


def test_generate_cli_delegates_to_pipeline(monkeypatch, tmp_path):
    import importlib

    cli = importlib.import_module("ai_dj.cli.main")

    class Result:
        output_path = tmp_path / "set.wav"
        report_path = tmp_path / "set.report.json"
        metrics = {"tracks_selected": 2, "transitions_generated": 1, "average_transition_score": 0.8, "technical_failures": 0}

    captured = {}

    def generate(directory, output, config, **kwargs):
        captured.update({"directory": directory, "output": output, "config": config, **kwargs})
        return Result()

    monkeypatch.setattr(cli, "generate_dj_set", generate)
    assert main(["generate", "--input", str(tmp_path), "--output", str(tmp_path / "set.wav"), "--tracks", "2", "--trajectory", "build"]) == 0
    assert captured["config"].target_track_count == 2
    assert captured["config"].energy_trajectory == "build"
