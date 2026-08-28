"""Simple, reproducible signal-energy descriptors."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.track import EnergyEstimate, EnergyPoint

DEFAULT_WINDOW_SECONDS = 1.0
_FLOOR_DB = -60.0


def estimate_energy(
    samples: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> EnergyEstimate:
    """Return normalized RMS level and a coarse one-value-per-window timeline.

    This is a level/intensity proxy, not a perceptual loudness or high-level
    musical-energy model. Keeping it separate makes it safe to replace later.
    """
    if samples.size == 0:
        return EnergyEstimate(global_level=0.0, timeline=())
    window_size = max(1, int(round(window_seconds * sample_rate)))
    points: list[EnergyPoint] = []
    for start in range(0, len(samples), window_size):
        window = samples[start : start + window_size]
        level = _normalized_rms(window)
        midpoint = (start + len(window) / 2) / sample_rate
        points.append(EnergyPoint(timestamp=round(float(midpoint), 6), value=round(level, 6)))
    return EnergyEstimate(global_level=round(_normalized_rms(samples), 6), timeline=tuple(points))


def _normalized_rms(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 1e-12:
        return 0.0
    db = 20.0 * np.log10(rms)
    return float(np.clip((db - _FLOOR_DB) / -_FLOOR_DB, 0.0, 1.0))
