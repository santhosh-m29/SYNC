"""Chroma-template estimation of Western key with explicit uncertainty."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.track import KeyEstimate

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(samples: np.ndarray, sample_rate: int) -> KeyEstimate:
    """Estimate a Western key from aggregate chroma using Krumhansl profiles.

    This deterministic estimate is intended as a preliminary harmonic feature.
    Modulations, atonal music, microtonal systems, and non-Western tonal systems
    can make the result unreliable; callers must use the returned confidence.
    """
    if samples.size == 0 or float(np.max(np.abs(samples))) <= 1e-8:
        return KeyEstimate(key=None, confidence=0.0)

    import librosa

    # Fixed equal-tempered bins avoid unstable tuning estimation on percussion.
    chroma = librosa.feature.chroma_stft(y=samples, sr=sample_rate, tuning=0.0)
    aggregate = np.mean(chroma, axis=1)
    if not np.isfinite(aggregate).all() or float(np.linalg.norm(aggregate)) <= 1e-12:
        return KeyEstimate(key=None, confidence=0.0)

    labels: list[str] = []
    scores: list[float] = []
    for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
        for tonic, name in enumerate(PITCH_CLASSES):
            labels.append(f"{name} {mode}")
            scores.append(_correlation(aggregate, np.roll(profile, tonic)))
    ordered = np.argsort(scores)[::-1]
    best = int(ordered[0])
    runner_up = int(ordered[1])
    best_score = max(0.0, float(scores[best]))
    separation = max(0.0, float(scores[best] - scores[runner_up]))
    confidence = float(np.clip(best_score * (0.5 + 0.5 * separation), 0.0, 1.0))
    return KeyEstimate(key=labels[best], confidence=round(confidence, 3))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    centered_left = left - np.mean(left)
    centered_right = right - np.mean(right)
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(centered_left, centered_right) / denominator)
