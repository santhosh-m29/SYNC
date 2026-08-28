"""Vocal-activity capability boundary for the local DSP pipeline."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.structure import VocalActivityEstimate


def estimate_vocal_activity(samples: np.ndarray, sample_rate: int) -> VocalActivityEstimate:
    """Return the current local vocal-analysis capability state.

    Librosa's generic DSP features cannot reliably distinguish singing from
    melodic instruments. Until a validated local vocal-activity model is added,
    return an explicit unavailable state instead of fabricated probabilities.
    """
    del samples, sample_rate
    return VocalActivityEstimate.unavailable()
