"""Beat timestamp estimation for normalized mono audio."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.track import BeatEstimate


def estimate_beats(samples: np.ndarray, sample_rate: int) -> BeatEstimate:
    """Return monotonically increasing beat positions in seconds and a confidence."""
    import librosa

    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=hop_length)
    if onset_envelope.size == 0 or float(np.max(onset_envelope)) <= 0.0:
        return BeatEstimate(timestamps=(), confidence=0.0)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sample_rate, hop_length=hop_length, trim=False, sparse=True
    )
    timestamps = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    valid = tuple(float(time) for time in timestamps if 0.0 <= time <= len(samples) / sample_rate)
    return BeatEstimate(timestamps=valid, confidence=round(_beat_confidence(onset_envelope, beat_frames), 3))


def _beat_confidence(onset_envelope: np.ndarray, beat_frames: np.ndarray) -> float:
    if len(beat_frames) < 2:
        return 0.0
    intervals = np.diff(beat_frames).astype(float)
    regularity = max(0.0, 1.0 - float(np.std(intervals) / max(np.mean(intervals), 1.0)))
    valid_frames = beat_frames[(beat_frames >= 0) & (beat_frames < len(onset_envelope))]
    if len(valid_frames) == 0:
        return 0.0
    frame_strength = float(np.mean(onset_envelope[valid_frames]) / max(float(np.max(onset_envelope)), 1e-12))
    return float(np.clip(0.5 * regularity + 0.5 * frame_strength, 0.0, 1.0))
