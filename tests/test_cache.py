from __future__ import annotations

import os

from ai_dj.pipeline.analyze import analyze_track
from ai_dj.pipeline.cache import AnalysisCache
from ai_dj.representation.track import ANALYSIS_VERSION
from tests.conftest import write_click_track


def test_cache_hit_and_file_modification_invalidation(tmp_path):
    source = write_click_track(tmp_path / "track.wav")
    cache = AnalysisCache(tmp_path / "cache")
    analysis = analyze_track(source)

    entry = cache.put(source, analysis)
    assert entry.is_file()
    assert cache.get(source) == analysis

    write_click_track(source, bpm=128.0)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert cache.get(source) is None


def test_cache_invalidates_when_analysis_version_changes(tmp_path):
    source = write_click_track(tmp_path / "track.wav")
    analysis = analyze_track(source)
    cache = AnalysisCache(tmp_path / "cache")
    cache.put(source, analysis)

    assert AnalysisCache(tmp_path / "cache", analysis_version=f"{ANALYSIS_VERSION}.next").get(source) is None
