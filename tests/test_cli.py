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
