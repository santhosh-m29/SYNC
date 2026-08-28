"""Compact global spectral/timbral descriptors for one track."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.track import SpectralFeatures

MFCC_COUNT = 13


def estimate_spectral_features(samples: np.ndarray, sample_rate: int) -> SpectralFeatures:
    """Compute stable summary statistics without retaining high-volume frames."""
    if samples.size == 0 or float(np.max(np.abs(samples))) <= 1e-8:
        return SpectralFeatures.zero()

    import librosa

    n_fft = 2048
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate, n_fft=n_fft)
    bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate, n_fft=n_fft)
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sample_rate, n_fft=n_fft)
    contrast = librosa.feature.spectral_contrast(y=samples, sr=sample_rate, n_fft=n_fft)
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=MFCC_COUNT, n_fft=n_fft)
    return SpectralFeatures(
        centroid_hz=_mean(centroid),
        bandwidth_hz=_mean(bandwidth),
        rolloff_hz=_mean(rolloff),
        contrast=tuple(round(float(value), 6) for value in np.mean(contrast, axis=1)),
        mfcc=tuple(round(float(value), 6) for value in np.mean(mfcc, axis=1)),
    )


def _mean(values: np.ndarray) -> float:
    return round(float(np.nan_to_num(np.mean(values), nan=0.0, posinf=0.0, neginf=0.0)), 6)
