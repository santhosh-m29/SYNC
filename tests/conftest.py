from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_click_track(
    path: Path,
    *,
    bpm: float = 120.0,
    duration: float = 8.0,
    sample_rate: int = 22_050,
) -> Path:
    """Create a deterministic mono PCM click track without copyrighted audio."""
    sample_count = int(duration * sample_rate)
    samples = np.zeros(sample_count, dtype=np.float32)
    click_length = int(0.015 * sample_rate)
    for start in range(0, sample_count, int(round(60.0 / bpm * sample_rate))):
        end = min(start + click_length, sample_count)
        samples[start:end] = np.hanning((end - start) * 2)[: end - start]
    pcm = np.clip(samples * 0.9, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((pcm * np.iinfo(np.int16).max).astype("<i2").tobytes())
    return path
