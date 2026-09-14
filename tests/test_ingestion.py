from __future__ import annotations

import pytest

from ai_dj.ingestion.loader import AudioLoadError, load_audio
from ai_dj.ingestion.scanner import AudioDirectoryNotFoundError, scan_audio_library
from tests.conftest import write_click_track


def test_scanner_finds_supported_files_recursively_and_skips_other_files(tmp_path):
    write_click_track(tmp_path / "root.wav")
    (tmp_path / "nested").mkdir()
    write_click_track(tmp_path / "nested" / "another.WAV")
    (tmp_path / "cover.jpg").write_bytes(b"not audio")
    (tmp_path / ".ai_dj_cache").mkdir()
    (tmp_path / ".ai_dj_cache" / "entry.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".ai_dj_stems" / "track").mkdir(parents=True)
    write_click_track(tmp_path / ".ai_dj_stems" / "track" / "vocals.wav")

    result = scan_audio_library(tmp_path)

    assert [path.name for path in result.audio_files] == ["another.WAV", "root.wav"]
    assert [path.name for path in result.skipped_files] == ["cover.jpg"]


def test_scanner_handles_missing_and_empty_directories(tmp_path):
    assert scan_audio_library(tmp_path).audio_files == ()
    with pytest.raises(AudioDirectoryNotFoundError):
        scan_audio_library(tmp_path / "missing")


def test_loader_returns_normalized_audio_and_duration(tmp_path):
    path = write_click_track(tmp_path / "valid.wav", duration=2.0, sample_rate=44_100)

    audio = load_audio(path, sample_rate=22_050)

    assert audio.samples.dtype.name == "float32"
    assert audio.samples.ndim == 1
    assert audio.sample_rate == 22_050
    assert audio.duration == pytest.approx(2.0, abs=0.01)


def test_loader_reports_corrupt_audio(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"this is not a WAV file")

    with pytest.raises(AudioLoadError):
        load_audio(broken)
