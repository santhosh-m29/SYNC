"""Tempo estimation with explicit confidence and octave normalization."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.track import TempoEstimate

MIN_TEMPO = 40.0
MAX_TEMPO = 240.0


def estimate_tempo(samples: np.ndarray, sample_rate: int) -> TempoEstimate:
    """Estimate a musically useful BPM; confidence expresses onset periodicity."""
    import librosa

    onset_envelope = librosa.onset.onset_strength(y=samples, sr=sample_rate)
    if onset_envelope.size == 0 or float(np.max(onset_envelope)) <= 0.0:
        return TempoEstimate(bpm=0.0, confidence=0.0)
    candidates = librosa.feature.tempo(onset_envelope=onset_envelope, sr=sample_rate, aggregate=None)
    raw_bpm = float(np.median(np.asarray(candidates)))
    bpm = _normalize_tempo_octave(raw_bpm)
    confidence = _periodicity_confidence(onset_envelope, sample_rate, bpm)
    alternatives = _octave_alternatives(bpm)
    return TempoEstimate(
        bpm=round(bpm, 3),
        confidence=round(confidence, 3),
        octave_alternatives=alternatives,
    )


def _normalize_tempo_octave(bpm: float) -> float:
    """Keep a plausible rhythmic tempo range while preserving ambiguity in confidence."""
    while 0 < bpm < MIN_TEMPO:
        bpm *= 2
    while bpm > MAX_TEMPO:
        bpm /= 2
    return bpm


def _octave_alternatives(bpm: float) -> tuple[float, ...]:
    """Expose plausible half/double-time readings without choosing one silently."""
    alternatives = {_normalize_tempo_octave(bpm / 2.0), _normalize_tempo_octave(bpm * 2.0)}
    alternatives.discard(bpm)
    return tuple(sorted(round(candidate, 3) for candidate in alternatives))


def _periodicity_confidence(onset_envelope: np.ndarray, sample_rate: int, bpm: float) -> float:
    if bpm <= 0 or onset_envelope.size < 4:
        return 0.0
    import librosa

    hop_length = 512
    lag = int(round(60.0 * sample_rate / (hop_length * bpm)))
    if lag <= 0 or lag >= onset_envelope.size:
        return 0.0
    centered = onset_envelope - np.mean(onset_envelope)
    denominator = float(np.linalg.norm(centered[:-lag]) * np.linalg.norm(centered[lag:]))
    if denominator <= 0:
        return 0.0
    correlation = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
    # Negative periodicity is no evidence; an octave-corrected correlation is still uncertain.
    return float(np.clip(max(0.0, correlation), 0.0, 1.0))
