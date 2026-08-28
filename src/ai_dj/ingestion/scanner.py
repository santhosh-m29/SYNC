"""Discovery of supported audio files without decoding them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = frozenset({".mp3", ".wav", ".flac"})
GENERATED_DIRECTORY_NAMES = frozenset({".ai_dj_cache", ".ai_dj_analysis"})


class AudioDirectoryNotFoundError(FileNotFoundError):
    """Raised when an audio-library root is absent or not a directory."""


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Supported files discovered during one library scan."""

    audio_files: tuple[Path, ...]
    skipped_files: tuple[Path, ...]


def scan_audio_files(directory: str | Path) -> list[Path]:
    """Return recursively discovered supported files in deterministic order."""
    return list(scan_audio_library(directory).audio_files)


def scan_audio_library(directory: str | Path) -> ScanResult:
    """Scan a directory recursively and retain unsupported-file diagnostics."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise AudioDirectoryNotFoundError(f"Audio directory does not exist: {root}")

    files: list[Path] = []
    skipped: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if GENERATED_DIRECTORY_NAMES.intersection(candidate.relative_to(root).parts):
            continue
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            LOGGER.debug("Found supported audio file: %s", candidate)
            files.append(candidate)
        else:
            LOGGER.debug("Skipping unsupported file: %s", candidate)
            skipped.append(candidate)
    return ScanResult(audio_files=tuple(files), skipped_files=tuple(skipped))
