"""Execute validated transition plans as lossless WAV previews.

This module never searches for tracks or transition points.  It receives two
analyzed tracks and an existing ``TransitionPlan`` and renders only that plan.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_dj.evaluation import assess_transition_constraints
from ai_dj.ingestion.loader import load_audio
from ai_dj.rendering.models import RenderConfig, RenderResult, StemPaths
from ai_dj.representation.track import TrackAnalysis
from ai_dj.transition.models import TransitionPlan

_SUPPORTED_STRATEGIES = {"phrase_crossfade", "instrumental_entry", "outro_intro"}


class RenderError(RuntimeError):
    """Raised when a plan cannot be safely executed as audio."""


def render_transition(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    plan: TransitionPlan,
    output_path: str | Path,
    config: RenderConfig = RenderConfig(),
    *,
    source_stems: StemPaths | None = None,
    destination_stems: StemPaths | None = None,
) -> RenderResult:
    """Render a plan-driven, pitch-preserving crossfade preview to float WAV.

    The preview begins with up to ``source_pre_roll_seconds`` before the source
    exit and ends with ``destination_post_roll_seconds`` after the overlap.
    """
    _validate_inputs(source, destination, plan, config)
    if (source_stems is None) != (destination_stems is None):
        raise RenderError("Vocal handoff requires trusted stems for both source and destination tracks")
    if source_stems is not None and destination_stems is not None:
        return _render_vocal_handoff(source, destination, plan, output_path, config, source_stems, destination_stems)
    source_audio = load_audio(source.source_path, sample_rate=config.sample_rate)
    destination_audio = load_audio(destination.source_path, sample_rate=config.sample_rate)
    _validate_audio_duration(source, destination, source_audio.duration, destination_audio.duration, plan)

    rate = _stretch_rate(source.tempo.bpm, destination.tempo.bpm)
    if abs(rate - 1.0) > config.maximum_tempo_adjustment:
        raise RenderError(f"Tempo stretch {rate:.3f} exceeds ±{config.maximum_tempo_adjustment:.0%} renderer limit")
    sample_rate = config.sample_rate
    overlap_samples = _sample_count(plan.duration, sample_rate)
    pre_samples = min(_sample_count(config.source_pre_roll_seconds, sample_rate), _sample_count(plan.source_exit, sample_rate))
    post_samples = _sample_count(config.destination_post_roll_seconds, sample_rate)
    source_start = _sample_count(plan.source_exit, sample_rate) - pre_samples
    source_segment = _slice_exact(source_audio.samples, source_start, pre_samples + overlap_samples)

    destination_input_length = _sample_count((plan.duration + config.destination_post_roll_seconds) * rate, sample_rate)
    destination_start = _sample_count(plan.destination_entry, sample_rate)
    destination_raw = _slice_exact(destination_audio.samples, destination_start, destination_input_length)
    destination_segment = _time_stretch(destination_raw, rate, overlap_samples + post_samples)
    alignment = _beat_alignment(source, destination, plan, rate, config.maximum_beat_alignment_seconds)
    destination_segment = _shift_fixed(destination_segment, _sample_count(alignment, sample_rate))

    source_overlap = source_segment[pre_samples:]
    destination_overlap = destination_segment[:overlap_samples]
    source_gain, destination_gain = _matched_gains(source_overlap, destination_overlap, config.maximum_gain_db)
    source_segment *= source_gain
    destination_segment *= destination_gain
    fade_out, fade_in = _fades(overlap_samples, plan.strategy)
    mix = source_segment[pre_samples:] * fade_out + destination_segment[:overlap_samples] * fade_in
    rendered = np.concatenate((source_segment[:pre_samples], mix, destination_segment[overlap_samples:]))
    rendered, protection_db = _peak_protect(rendered, config.peak_ceiling)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, rendered, sample_rate, format="WAV", subtype="FLOAT")
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(rendered)))) if rendered.size else 0.0
    return RenderResult(
        output_path=output,
        sample_rate=sample_rate,
        duration=round(rendered.size / sample_rate, 6),
        transition_start=round(pre_samples / sample_rate, 6),
        transition_duration=plan.duration,
        strategy=plan.strategy,
        time_stretch_rate=round(rate, 6),
        beat_alignment_seconds=round(alignment, 6),
        source_gain_db=round(_gain_db(source_gain), 6),
        destination_gain_db=round(_gain_db(destination_gain), 6),
        peak_protection_db=round(protection_db, 6),
        peak=round(peak, 6),
        rms=round(rms, 6),
        clipping_samples=int(np.count_nonzero(np.abs(rendered) > 1.0)),
    )


def _render_vocal_handoff(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    plan: TransitionPlan,
    output_path: str | Path,
    config: RenderConfig,
    source_stems: StemPaths,
    destination_stems: StemPaths,
) -> RenderResult:
    """Render a stem-only vocal handoff; instrumental layers retain normal mix flow."""
    sample_rate = config.sample_rate
    source_instrumental = load_audio(source_stems.instrumental_path, sample_rate=sample_rate)
    source_vocals = load_audio(source_stems.vocal_path, sample_rate=sample_rate)
    destination_instrumental = load_audio(destination_stems.instrumental_path, sample_rate=sample_rate)
    destination_vocals = load_audio(destination_stems.vocal_path, sample_rate=sample_rate)
    for duration in (source_instrumental.duration, source_vocals.duration):
        _validate_audio_duration(source, destination, duration, destination_instrumental.duration, plan)
    for duration in (destination_instrumental.duration, destination_vocals.duration):
        _validate_audio_duration(source, destination, source_instrumental.duration, duration, plan)

    rate = _stretch_rate(source.tempo.bpm, destination.tempo.bpm)
    if abs(rate - 1.0) > config.maximum_tempo_adjustment:
        raise RenderError(f"Tempo stretch {rate:.3f} exceeds ±{config.maximum_tempo_adjustment:.0%} renderer limit")
    overlap = _sample_count(plan.duration, sample_rate)
    pre = min(_sample_count(config.source_pre_roll_seconds, sample_rate), _sample_count(plan.source_exit, sample_rate))
    post = _sample_count(config.destination_post_roll_seconds, sample_rate)
    source_start = _sample_count(plan.source_exit, sample_rate) - pre
    source_instrumental_segment = _slice_exact(source_instrumental.samples, source_start, pre + overlap)
    source_vocal_segment = _slice_exact(source_vocals.samples, source_start, pre + overlap)
    destination_start = _sample_count(plan.destination_entry, sample_rate)
    destination_input_length = _sample_count((plan.duration + config.destination_post_roll_seconds) * rate, sample_rate)
    destination_instrumental_segment = _time_stretch(
        _slice_exact(destination_instrumental.samples, destination_start, destination_input_length), rate, overlap + post
    )
    destination_vocal_segment = _time_stretch(
        _slice_exact(destination_vocals.samples, destination_start, destination_input_length), rate, overlap + post
    )
    alignment = _beat_alignment(source, destination, plan, rate, config.maximum_beat_alignment_seconds)
    shift = _sample_count(alignment, sample_rate)
    destination_instrumental_segment = _shift_fixed(destination_instrumental_segment, shift)
    destination_vocal_segment = _shift_fixed(destination_vocal_segment, shift)
    source_gain, destination_gain = _matched_gains(
        source_instrumental_segment[pre:], destination_instrumental_segment[:overlap], config.maximum_gain_db
    )
    source_instrumental_segment *= source_gain
    source_vocal_segment *= source_gain
    destination_instrumental_segment *= destination_gain
    destination_vocal_segment *= destination_gain
    fade_out, fade_in = _fades(overlap, plan.strategy)
    source_vocal_envelope, destination_vocal_envelope, entry = _vocal_handoff_envelopes(overlap, config)
    mixed_overlap = (
        source_instrumental_segment[pre:] * fade_out
        + destination_instrumental_segment[:overlap] * fade_in
        + source_vocal_segment[pre:] * source_vocal_envelope
        + destination_vocal_segment[:overlap] * destination_vocal_envelope
    )
    rendered = np.concatenate(
        (
            source_instrumental_segment[:pre] + source_vocal_segment[:pre],
            mixed_overlap,
            destination_instrumental_segment[overlap:] + destination_vocal_segment[overlap:],
        )
    )
    rendered, protection_db = _peak_protect(rendered, config.peak_ceiling)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, rendered, sample_rate, format="WAV", subtype="FLOAT")
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(rendered)))) if rendered.size else 0.0
    return RenderResult(
        output_path=output,
        sample_rate=sample_rate,
        duration=round(rendered.size / sample_rate, 6),
        transition_start=round(pre / sample_rate, 6),
        transition_duration=plan.duration,
        strategy=plan.strategy,
        time_stretch_rate=round(rate, 6),
        beat_alignment_seconds=round(alignment, 6),
        source_gain_db=round(_gain_db(source_gain), 6),
        destination_gain_db=round(_gain_db(destination_gain), 6),
        peak_protection_db=round(protection_db, 6),
        peak=round(peak, 6),
        rms=round(rms, 6),
        clipping_samples=int(np.count_nonzero(np.abs(rendered) > 1.0)),
        vocal_handoff_applied=True,
        destination_vocal_entry_seconds=round(entry / sample_rate, 6),
    )


def _validate_inputs(source: TrackAnalysis, destination: TrackAnalysis, plan: TransitionPlan, config: RenderConfig) -> None:
    if plan.strategy not in _SUPPORTED_STRATEGIES:
        raise RenderError(f"Unsupported render strategy: {plan.strategy}")
    constraint = assess_transition_constraints(source, destination, plan)
    if not constraint.allowed:
        raise RenderError("Transition plan is ineligible: " + "; ".join(constraint.reasons))
    if config.sample_rate <= 0 or config.peak_ceiling <= 0.0 or config.peak_ceiling > 1.0:
        raise ValueError("Invalid renderer sample rate or peak ceiling")
    if not (0.0 < config.source_vocal_release_fraction < 1.0 and 0.0 < config.destination_vocal_entry_fraction < 1.0):
        raise ValueError("Vocal handoff fractions must be between zero and one")


def _validate_audio_duration(source: TrackAnalysis, destination: TrackAnalysis, source_duration: float, destination_duration: float, plan: TransitionPlan) -> None:
    if plan.source_exit + plan.duration > source_duration + 1e-3:
        raise RenderError("Source audio is shorter than the planned transition")
    if plan.destination_entry + plan.duration > destination_duration + 1e-3:
        raise RenderError("Destination audio is shorter than the planned transition")


def _stretch_rate(source_bpm: float, destination_bpm: float) -> float:
    ratio = source_bpm / destination_bpm
    relationship = min((0.5, 1.0, 2.0), key=lambda value: abs(ratio / value - 1.0))
    return ratio / relationship


def _time_stretch(samples: np.ndarray, rate: float, expected_length: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(expected_length, dtype=np.float32)
    if math.isclose(rate, 1.0, abs_tol=1e-6):
        stretched = samples
    else:
        import librosa

        stretched = librosa.effects.time_stretch(samples.astype(np.float32, copy=False), rate=rate)
    return _fixed_length(np.asarray(stretched, dtype=np.float32), expected_length)


def _beat_alignment(source: TrackAnalysis, destination: TrackAnalysis, plan: TransitionPlan, rate: float, maximum: float) -> float:
    source_beat = _nearest(source.beats.timestamps, plan.source_exit)
    destination_beat = _nearest(destination.beats.timestamps, plan.destination_entry)
    if source_beat is None or destination_beat is None:
        return 0.0
    source_offset = source_beat - plan.source_exit
    destination_offset = (destination_beat - plan.destination_entry) / rate
    return float(np.clip(source_offset - destination_offset, -maximum, maximum))


def _nearest(values: tuple[float, ...], target: float) -> float | None:
    return min(values, key=lambda value: abs(value - target)) if values else None


def _matched_gains(source: np.ndarray, destination: np.ndarray, maximum_db: float) -> tuple[float, float]:
    source_rms = max(float(np.sqrt(np.mean(source**2))), 1e-6)
    destination_rms = max(float(np.sqrt(np.mean(destination**2))), 1e-6)
    destination_db = float(np.clip(20.0 * np.log10(source_rms / destination_rms), -maximum_db, maximum_db))
    return 1.0, float(10.0 ** (destination_db / 20.0))


def _fades(length: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, np.pi / 2.0, length, endpoint=True, dtype=np.float32)
    if strategy == "instrumental_entry":
        incoming = np.sin(phase) ** 1.5
        return np.sqrt(1.0 - incoming**2), incoming
    return np.cos(phase), np.sin(phase)


def _vocal_handoff_envelopes(length: int, config: RenderConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """Fade source vocals first, then introduce destination vocals alone."""
    release_end = max(1, min(length, int(round(length * config.source_vocal_release_fraction))))
    entry_start = max(release_end, min(length - 1, int(round(length * config.destination_vocal_entry_fraction))))
    source = np.zeros(length, dtype=np.float32)
    source[:release_end] = np.cos(np.linspace(0.0, np.pi / 2.0, release_end, endpoint=True, dtype=np.float32))
    destination = np.zeros(length, dtype=np.float32)
    destination[entry_start:] = np.sin(
        np.linspace(0.0, np.pi / 2.0, length - entry_start, endpoint=True, dtype=np.float32)
    )
    return source, destination, entry_start


def _peak_protect(samples: np.ndarray, ceiling: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= ceiling or peak == 0.0:
        return samples.astype(np.float32, copy=False), 0.0
    gain = ceiling / peak
    return (samples * gain).astype(np.float32), _gain_db(gain)


def _slice_exact(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    if start < 0 or length < 0:
        raise RenderError("Negative audio slice requested")
    return _fixed_length(samples[start : start + length], length)


def _shift_fixed(samples: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return samples
    if offset > 0:
        return _fixed_length(np.concatenate((np.zeros(offset, dtype=np.float32), samples)), samples.size)
    return _fixed_length(samples[-offset:], samples.size)


def _fixed_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.size >= length:
        return np.ascontiguousarray(samples[:length], dtype=np.float32)
    return np.pad(samples, (0, length - samples.size)).astype(np.float32, copy=False)


def _sample_count(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def _gain_db(gain: float) -> float:
    return 20.0 * math.log10(max(gain, 1e-12))
