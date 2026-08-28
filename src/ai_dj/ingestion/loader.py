"""One-track-at-a-time audio decoding and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioLoadError(RuntimeError):
    """Raised when a source cannot be decoded into normalized audio."""


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int
    duration: float


def load_audio(path: str | Path, *, sample_rate: int = 22_050) -> AudioBuffer:
    """Decode one file, downmix to mono, resample, and return float32 samples."""
    source = Path(path)
    try:
        samples, original_rate = sf.read(source, dtype="float32", always_2d=True)
        mono = np.mean(samples, axis=1, dtype=np.float32)
        normalized = _resample(mono, original_rate, sample_rate)
    except Exception as soundfile_error:
        # librosa/audioread offers a useful fallback for MP3 backend variation.
        try:
            import librosa

            normalized, loaded_rate = librosa.load(source, sr=sample_rate, mono=True, dtype=np.float32)
            if loaded_rate != sample_rate:
                raise AudioLoadError(f"Unexpected decoded rate {loaded_rate} for {source}")
        except Exception as fallback_error:
            raise AudioLoadError(f"Could not decode {source}: {fallback_error}") from soundfile_error

    if normalized.size == 0:
        raise AudioLoadError(f"Audio file contains no samples: {source}")
    if not np.isfinite(normalized).all():
        raise AudioLoadError(f"Audio file contains non-finite samples: {source}")
    return AudioBuffer(samples=np.ascontiguousarray(normalized, dtype=np.float32), sample_rate=sample_rate,
                       duration=float(normalized.size / sample_rate))


def _resample(samples: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return samples
    import librosa

    return librosa.resample(samples, orig_sr=original_rate, target_sr=target_rate).astype(np.float32, copy=False)
