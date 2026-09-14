"""Vocal-activity capability boundary for the local DSP pipeline."""

from __future__ import annotations

import numpy as np

from ai_dj.representation.structure import VocalActivityEstimate
from ai_dj.representation.structure import VocalActivity


def estimate_vocal_activity(samples: np.ndarray, sample_rate: int) -> VocalActivityEstimate:
    """Return the current local vocal-analysis capability state.

    Librosa's generic DSP features cannot reliably distinguish singing from
    melodic instruments. Until a validated local vocal-activity model is added,
    return an explicit unavailable state instead of fabricated probabilities.
    """
    del samples, sample_rate
    return VocalActivityEstimate.unavailable()


def timeline_from_stems(vocals, accompaniment, sample_rate):
    """Measure activity in a *separated vocal source*, never in the mixed song.

    Hysteresis, 200 ms context and a 300 ms release retain quiet syllables and
    reverberant endings. Values are activity indicators, not calibrated model
    probabilities. Separation leakage remains possible, so confidence stays
    below the planner's trust threshold even when the stem sounds clear.
    """
    if sample_rate <= 0 or len(vocals) != len(accompaniment) or not len(vocals):
        raise ValueError("Aligned nonempty stems and a positive sample rate are required")
    if not np.isfinite(vocals).all() or not np.isfinite(accompaniment).all():
        raise ValueError("Nonfinite stem audio")
    vocals = np.asarray(vocals)
    accompaniment = np.asarray(accompaniment)
    hop = max(1, round(.05 * sample_rate))
    def energy(samples):
        if samples.ndim == 2:
            samples = np.mean(samples ** 2, axis=1)
        else:
            samples = samples ** 2
        padded = np.pad(samples, (0, (-len(samples)) % hop))
        return np.sqrt(padded.reshape(-1, hop).mean(axis=1))
    voice, backing = energy(vocals), energy(accompaniment)
    db = 20 * np.log10(np.maximum(voice, 1e-8))
    relative = 20 * np.log10(np.maximum(voice, 1e-8) / np.maximum(backing, 1e-8))
    # An absolute noise floor prevents silent stems from becoming "vocals" via normalization.
    on = (db > -48) & (relative > -24)
    hold = (db > -54) & (relative > -30)
    active = np.zeros(len(on), dtype=bool)
    release = 0
    for i in range(len(on)):
        if on[i] or (release > 0 and hold[i]):
            release = 6
        active[i] = release > 0
        release = max(0, release - 1)
    # Include attack context to avoid cueing after a syllable has already started.
    padded = active.copy()
    for offset in range(1, 5):
        padded[:-offset] |= active[offset:]
    edges = np.flatnonzero(np.diff(np.r_[False, padded, False].astype(int)))
    duration = len(vocals) / sample_rate
    segments = tuple(VocalActivity(start * hop / sample_rate, min(end * hop / sample_rate, duration), 1.0)
                     for start, end in zip(edges[::2], edges[1::2]))
    return VocalActivityEstimate(segments, True, "demucs_stem_activity_v1_uncalibrated", .5)


def analyze_track_vocals(source_path, cache_directory):
    """Reuse the existing local Demucs cache, with a versioned activity sidecar."""
    import json
    import logging
    import soundfile as sf
    from ai_dj.rendering.stems import separate_stems
    try:
        stems = separate_stems(source_path, cache_directory)
        sidecar = stems.vocals.parent / "activity-v1.json"
        if sidecar.is_file():
            try:
                return VocalActivityEstimate.from_dict(json.loads(sidecar.read_text()))
            except (ValueError, KeyError, TypeError):
                pass
        voice, sr = sf.read(stems.vocals, dtype="float32", always_2d=True)
        backing, backing_sr = sf.read(stems.accompaniment, dtype="float32", always_2d=True)
        if sr != backing_sr:
            raise ValueError("Stem sample rates differ")
        result = timeline_from_stems(voice, backing, sr)
        temporary = sidecar.with_suffix(".tmp")
        temporary.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        temporary.replace(sidecar)
        return result
    except Exception as error:
        logging.getLogger(__name__).warning("Vocal detection unavailable for %s: %s", source_path, error)
        return VocalActivityEstimate.unavailable()


def read_cached_track_vocals(source_path, cache_directory):
    """Read completed evidence without importing a model or triggering separation."""
    import json
    from ai_dj.rendering.stems import stem_cache_root
    try:
        root = stem_cache_root(source_path, cache_directory)
        if not (root / "complete-v2").is_file():
            return None
        return VocalActivityEstimate.from_dict(json.loads((root / "activity-v1.json").read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return None
