"""Heuristic downbeat inference from beat positions and onset accents."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.track import BeatEstimate, DownbeatEstimate

SUPPORTED_METERS = (3, 4)


def estimate_downbeats(
    samples: np.ndarray,
    sample_rate: int,
    beats: BeatEstimate,
) -> DownbeatEstimate:
    """Infer accented beat positions for common triple and quadruple meters.

    This is intentionally a confidence-scored heuristic, not a time-signature
    classifier. Tracks in other or changing meters may produce low-confidence
    or incorrect results and remain candidates for a later structure analyzer.
    """
    if len(beats.timestamps) < 3:
        return DownbeatEstimate(timestamps=(), confidence=0.0, meter=None)

    import librosa

    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=hop_length)
    frames = librosa.time_to_frames(np.asarray(beats.timestamps), sr=sample_rate, hop_length=hop_length)
    valid = (frames >= 0) & (frames < len(onset_envelope))
    frames = frames[valid]
    timestamps = np.asarray(beats.timestamps)[valid]
    if len(frames) < 3:
        return DownbeatEstimate(timestamps=(), confidence=0.0, meter=None)

    strengths = onset_envelope[frames]
    meter, phase, confidence = _select_meter_and_phase(strengths)
    downbeats = tuple(float(value) for value in timestamps[phase::meter])
    return DownbeatEstimate(timestamps=downbeats, confidence=round(confidence, 3), meter=meter)


def _select_meter_and_phase(strengths: np.ndarray) -> tuple[int, int, float]:
    best: tuple[float, int, int, float] | None = None
    for meter in SUPPORTED_METERS:
        if len(strengths) < meter:
            continue
        scores = np.array([np.mean(strengths[phase::meter]) for phase in range(meter)])
        phase = int(np.argmax(scores))
        accent_strength = float(scores[phase])
        phase_contrast = max(0.0, accent_strength - float(np.mean(scores)))
        confidence = phase_contrast / max(accent_strength, 1e-12)
        candidate = (accent_strength, meter, phase, confidence)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return 4, 0, 0.0
    _, meter, phase, confidence = best
    return meter, phase, float(np.clip(confidence, 0.0, 1.0))
