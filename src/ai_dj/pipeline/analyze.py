"""Orchestration layer joining ingestion, DSP analysis, representation, and cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ai_dj.analysis.beats import estimate_beats
from ai_dj.analysis.downbeats import estimate_downbeats
from ai_dj.analysis.energy import estimate_energy
from ai_dj.analysis.key import estimate_key
from ai_dj.analysis.spectral import estimate_spectral_features
from ai_dj.analysis.tempo import estimate_tempo
from ai_dj.ingestion.loader import load_audio
from ai_dj.ingestion.scanner import scan_audio_library
from ai_dj.pipeline.cache import AnalysisCache
from ai_dj.representation.track import ANALYSIS_VERSION, TrackAnalysis

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LibraryAnalysisResult:
    analyses: list[TrackAnalysis]
    failures: list[tuple[Path, Exception]]
    cached: int
    skipped: int


def analyze_track(source: str | Path) -> TrackAnalysis:
    path = Path(source)
    audio = load_audio(path)
    tempo = estimate_tempo(audio.samples, audio.sample_rate)
    beats = estimate_beats(audio.samples, audio.sample_rate)
    return TrackAnalysis(
        track_id=TrackAnalysis.track_id_for(path),
        source_path=str(path.resolve()),
        duration=round(audio.duration, 6),
        tempo=tempo,
        beats=beats,
        downbeats=estimate_downbeats(audio.samples, audio.sample_rate, beats),
        key=estimate_key(audio.samples, audio.sample_rate),
        energy=estimate_energy(audio.samples, audio.sample_rate),
        spectral=estimate_spectral_features(audio.samples, audio.sample_rate),
        analysis_version=ANALYSIS_VERSION,
    )


def analyze_library(directory: str | Path, cache: AnalysisCache) -> LibraryAnalysisResult:
    analyses: list[TrackAnalysis] = []
    failures: list[tuple[Path, Exception]] = []
    cached = 0
    scan = scan_audio_library(directory)
    tracks = scan.audio_files
    LOGGER.info("Found %d supported audio files", len(tracks))
    if scan.skipped_files:
        LOGGER.info("Skipped %d unsupported files", len(scan.skipped_files))
    for path in tracks:
        try:
            existing = cache.get(path)
            if existing is not None:
                LOGGER.info("Using cached analysis: %s", path.name)
                analyses.append(existing)
                cached += 1
                continue
            LOGGER.info("Analyzing %s", path.name)
            analysis = analyze_track(path)
            cache.put(path, analysis)
            LOGGER.info(
                "BPM: %.3f; beats: %d; downbeats: %d; key: %s; energy: %.3f",
                analysis.tempo.bpm,
                len(analysis.beats.timestamps),
                len(analysis.downbeats.timestamps),
                analysis.key.key or "unknown",
                analysis.energy.global_level,
            )
            analyses.append(analysis)
        except Exception as error:
            LOGGER.warning("Failed to analyze %s: %s", path.name, error)
            failures.append((path, error))
    return LibraryAnalysisResult(
        analyses=analyses,
        failures=failures,
        cached=cached,
        skipped=len(scan.skipped_files),
    )
