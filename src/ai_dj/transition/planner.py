"""Candidate transition search using analyzed musical positions only.

The planner does not alter audio. It combines Phase 3 track-pair compatibility
with local scores at bar/phrase/section/downbeat-derived entry and exit points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_dj.matching import score_track_pair
from ai_dj.matching.models import CompatibilityResult
from ai_dj.representation.structure import VocalActivityEstimate
from ai_dj.representation.track import TrackAnalysis
from ai_dj.transition.models import TransitionComponent, TransitionPlan
from ai_dj.transition.vocal_safety import assess_vocal_safety

TRANSITION_WEIGHTS = {
    "tempo": 0.15,
    "beat_alignment": 0.15,
    "phrase_alignment": 0.20,
    "harmony": 0.15,
    "energy": 0.15,
    "vocals": 0.20,
    "structure": 0.10,
}
_DURATIONS = (16.0, 8.0, 4.0)
_MIN_DURATION = min(_DURATIONS)


@dataclass(frozen=True, slots=True)
class _CandidatePoint:
    timestamp: float
    kinds: frozenset[str]
    confidence: float


def find_best_transitions(
    source: TrackAnalysis, destination: TrackAnalysis, *, include_rejected: bool = False,
    search_durations: bool = False,
) -> list[TransitionPlan]:
    """Return valid source-exit × destination-entry candidates in ranked order."""
    pair = score_track_pair(source, destination)
    exits = _exit_points(source)
    entries = _entry_points(destination)
    plans: list[TransitionPlan] = []
    for exit_point in exits:
        for entry_point in entries:
            duration = _duration_for(source, destination, exit_point.timestamp, entry_point.timestamp)
            if duration is None:
                continue
            durations = [d for d in _DURATIONS if d <= duration] if search_durations else [duration]
            for candidate_duration in durations:
                plans.append(_plan(source, destination, pair, exit_point, entry_point, candidate_duration))
    ranked = sorted(
        plans,
        key=lambda plan: (-plan.overall_score, -plan.confidence, -plan.duration, plan.source_exit, plan.destination_entry),
    )
    # Dataset builders can retain rejected overlaps as hard negatives. Playback
    # receives only safe crossfades. If no crossfade is provably safe, use a
    # beat/phrase-aligned zero-overlap handoff: it preserves the no-simultaneous-
    # vocals rule without treating an entire vocal song as ineligible.
    if include_rejected:
        return ranked
    safe = [plan for plan in ranked if plan.vocal_safety == "safe"]
    if safe:
        return safe
    eligible = [plan for plan in ranked if plan.vocal_safety != "reject"]
    return eligible or _handoff_plans(source, destination, pair, exits, entries)


def find_best_transition(source: TrackAnalysis, destination: TrackAnalysis) -> TransitionPlan | None:
    """Return the strongest valid plan, or ``None`` when evidence yields none."""
    candidates = find_best_transitions(source, destination)
    return candidates[0] if candidates else None


def _exit_points(track: TrackAnalysis) -> tuple[_CandidatePoint, ...]:
    points: dict[float, _CandidatePoint] = {}
    latest_start = track.duration - _MIN_DURATION
    # Search the full analysed structure, but keep a meaningful early exit range
    # available. Ranking below prefers a completed musical block over the final
    # repetition, so this is not a fixed timestamp rule.
    for phrase in track.structure.phrases:
        _add_point(points, phrase.start, "phrase", phrase.confidence, track.duration, lower=track.duration * 0.25, upper=latest_start)
    for section in track.structure.sections:
        # Exiting at the end of an analysed section preserves the complete hook
        # or verse; section starts remain useful as incoming entry points.
        _add_point(points, section.end, "section", section.confidence, track.duration, lower=track.duration * 0.30, upper=latest_start)
    for bar in track.structure.bars:
        _add_point(points, bar.start, "bar", bar.confidence, track.duration, lower=track.duration * 0.35, upper=latest_start)
    for timestamp in track.downbeats.timestamps:
        _add_point(points, timestamp, "downbeat", track.downbeats.confidence, track.duration, lower=track.duration * 0.40, upper=latest_start)
    return _cap_points(tuple(points[key] for key in sorted(points)))


def _entry_points(track: TrackAnalysis) -> tuple[_CandidatePoint, ...]:
    points: dict[float, _CandidatePoint] = {}
    latest_start = track.duration - _MIN_DURATION
    # Entry points may be later sections: a strong instrumental/buildup or hook
    # can be a better musical entrance than an arbitrary 0:00 intro.
    for phrase in track.structure.phrases:
        _add_point(points, phrase.start, "phrase", phrase.confidence, track.duration, lower=0.0, upper=min(track.duration * 0.80, latest_start))
    for section in track.structure.sections:
        _add_point(points, section.start, "section", section.confidence, track.duration, lower=0.0, upper=min(track.duration * 0.80, latest_start))
    for bar in track.structure.bars:
        _add_point(points, bar.start, "bar", bar.confidence, track.duration, lower=0.0, upper=min(track.duration * 0.65, latest_start))
    for timestamp in track.downbeats.timestamps:
        _add_point(points, timestamp, "downbeat", track.downbeats.confidence, track.duration, lower=0.0, upper=min(track.duration * 0.55, latest_start))
    return _cap_points(tuple(points[key] for key in sorted(points)))


def _cap_points(points: tuple[_CandidatePoint, ...], limit: int = 16) -> tuple[_CandidatePoint, ...]:
    """Keep the search bounded while preserving the whole musical timeline."""
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=int)
    return tuple(points[int(index)] for index in sorted(set(indices)))


def _add_point(
    points: dict[float, _CandidatePoint],
    timestamp: float,
    kind: str,
    confidence: float,
    duration: float,
    *,
    lower: float,
    upper: float,
) -> None:
    if not (lower <= timestamp <= upper < duration):
        return
    key = round(float(timestamp), 6)
    existing = points.get(key)
    if existing is None:
        points[key] = _CandidatePoint(key, frozenset({kind}), float(np.clip(confidence, 0.0, 1.0)))
    else:
        points[key] = _CandidatePoint(
            key,
            existing.kinds | {kind},
            max(existing.confidence, float(np.clip(confidence, 0.0, 1.0))),
        )


def _duration_for(source: TrackAnalysis, destination: TrackAnalysis, source_exit: float, destination_entry: float) -> float | None:
    for duration in _DURATIONS:
        if source_exit + duration <= source.duration and destination_entry + duration <= destination.duration:
            return duration
    return None


def _handoff_plans(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    pair: CompatibilityResult,
    exits: tuple[_CandidatePoint, ...],
    entries: tuple[_CandidatePoint, ...],
) -> list[TransitionPlan]:
    """Create a last-resort no-overlap handoff at existing musical boundaries."""
    plans = [_plan(source, destination, pair, exit_point, entry_point, 0.0) for exit_point in exits for entry_point in entries]
    return sorted(
        plans,
        key=lambda plan: (-plan.overall_score, -plan.confidence, plan.source_exit, plan.destination_entry),
    )


def _plan(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    pair: CompatibilityResult,
    exit_point: _CandidatePoint,
    entry_point: _CandidatePoint,
    duration: float,
) -> TransitionPlan:
    components = (
        _pair_component(pair, "tempo"),
        _beat_alignment_component(source, destination, exit_point, entry_point),
        _phrase_component(exit_point, entry_point),
        _pair_component(pair, "harmony"),
        _energy_component(source, destination, exit_point.timestamp, entry_point.timestamp),
        _vocal_component(source, destination, exit_point.timestamp, entry_point.timestamp, duration),
        _structure_component(pair, exit_point, entry_point),
    )
    effective_weight = sum(component.weight * component.confidence for component in components)
    base_weight = sum(component.weight for component in components)
    overall = (
        sum(component.score * component.weight * component.confidence for component in components) / effective_weight
        if effective_weight > 0
        else 0.5
    )
    # Prefer a completed musical block in the main body over the final repeat.
    # This is a soft ranking term; a weak early boundary still loses to a strong
    # later phrase when the compatibility evidence warrants it.
    exit_fraction = exit_point.timestamp / max(source.duration, 1e-9)
    if 0.40 <= exit_fraction <= 0.78:
        overall += 0.045
    elif exit_fraction > 0.88:
        overall -= 0.08 * (exit_fraction - 0.88) / 0.12
    vocal = next(component for component in components if component.name == "vocals")
    provisional = TransitionPlan(
        source_track_id=source.track_id,
        destination_track_id=destination.track_id,
        source_exit=exit_point.timestamp,
        destination_entry=entry_point.timestamp,
        duration=duration,
        strategy=_strategy(source, destination, exit_point, entry_point, duration),
        overall_score=0.0,
        confidence=0.0,
        components=(),
        strengths=(),
        weaknesses=(),
    )
    safety = assess_vocal_safety(source, destination, provisional)
    vocal_safety = "safe" if safety.allowed else "reject" if safety.verified else "unverifiable"
    # This explicit penalty makes a proven vocal collision lose to a safe
    # structural alternative even when its tempo/key score is slightly higher.
    if vocal.confidence > 0.0 and vocal_safety == "reject":
        overall *= 0.55 + 0.45 * vocal.score
    strengths = tuple(component.reason for component in components if component.confidence >= 0.5 and component.score >= 0.75)
    weaknesses = tuple(component.reason for component in components if component.confidence >= 0.5 and component.score < 0.45)
    return TransitionPlan(
        source_track_id=source.track_id,
        destination_track_id=destination.track_id,
        source_exit=exit_point.timestamp,
        destination_entry=entry_point.timestamp,
        duration=duration,
        strategy=provisional.strategy,
        overall_score=round(float(np.clip(overall, 0.0, 1.0)), 3),
        confidence=round(float(np.clip(effective_weight / base_weight, 0.0, 1.0)), 3),
        components=components,
        strengths=strengths,
        weaknesses=weaknesses,
        incoming_vocal_start=_incoming_vocal_start(destination, entry_point.timestamp),
        vocal_safety=vocal_safety,
        vocal_collision_duration=safety.collision_duration,
        maximum_vocal_overlap_probability=safety.maximum_overlap_probability,
        integrated_vocal_overlap=safety.integrated_overlap,
    )


def _pair_component(pair: CompatibilityResult, name: str) -> TransitionComponent:
    component = next(item for item in pair.components if item.name == name)
    return _component(name, component.score, component.confidence, component.reason)


def _beat_alignment_component(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    exit_point: _CandidatePoint,
    entry_point: _CandidatePoint,
) -> TransitionComponent:
    both_downbeats = "downbeat" in exit_point.kinds and "downbeat" in entry_point.kinds
    meter_match = source.downbeats.meter is not None and source.downbeats.meter == destination.downbeats.meter
    score = 1.0 if both_downbeats and meter_match else 0.85 if both_downbeats else 0.7
    confidence = min(exit_point.confidence, entry_point.confidence, source.beats.confidence, destination.beats.confidence)
    return _component("beat_alignment", score, confidence, "Downbeat-aligned entry/exit" if both_downbeats else "Bar/phrase-aligned entry/exit")


def _phrase_component(exit_point: _CandidatePoint, entry_point: _CandidatePoint) -> TransitionComponent:
    both_phrases = "phrase" in exit_point.kinds and "phrase" in entry_point.kinds
    score = 1.0 if both_phrases else 0.75 if "phrase" in exit_point.kinds or "phrase" in entry_point.kinds else 0.55
    return _component("phrase_alignment", score, min(exit_point.confidence, entry_point.confidence), "Phrase-aligned transition" if both_phrases else "Partial phrase alignment")


def _energy_component(source: TrackAnalysis, destination: TrackAnalysis, source_time: float, destination_time: float) -> TransitionComponent:
    source_level = _energy_at(source, source_time)
    destination_level = _energy_at(destination, destination_time)
    score = float(np.clip(1.0 - abs(source_level - destination_level) / 0.5, 0.0, 1.0))
    confidence = 1.0 if source.energy.timeline and destination.energy.timeline else 0.5
    return _component("energy", score, confidence, "Local energy continuity")


def _vocal_component(source: TrackAnalysis, destination: TrackAnalysis, source_time: float, destination_time: float, duration: float) -> TransitionComponent:
    source_vocals = source.structure.vocal_activity
    destination_vocals = destination.structure.vocal_activity
    if not source_vocals.available or not destination_vocals.available:
        return _component("vocals", 0.5, 0.0, "Vocal activity unavailable")
    collision = _vocal_overlap(source_vocals, destination_vocals, source_time, destination_time, duration)
    source_probability = _vocal_probability(source_vocals, source_time, source_time + duration)
    destination_probability = _vocal_probability(destination_vocals, destination_time, destination_time + duration)
    return _component(
        "vocals",
        1.0 - collision,
        1.0,
        f"Vocal collision integral {collision:.3f}; source {source_probability:.3f}, destination {destination_probability:.3f}",
    )


def _structure_component(pair: CompatibilityResult, exit_point: _CandidatePoint, entry_point: _CandidatePoint) -> TransitionComponent:
    pair_structure = next(item for item in pair.components if item.name == "structure")
    local_score = 1.0 if "section" in exit_point.kinds and "section" in entry_point.kinds else 0.8
    return _component(
        "structure",
        0.5 * pair_structure.score + 0.5 * local_score,
        min(pair_structure.confidence, exit_point.confidence, entry_point.confidence),
        "Structural boundary candidate compatibility",
    )


def _strategy(
    source: TrackAnalysis, destination: TrackAnalysis, exit_point: _CandidatePoint, entry_point: _CandidatePoint, duration: float
) -> str:
    if duration == 0.0:
        return "hard_handoff"
    incoming_start = _incoming_vocal_start(destination, entry_point.timestamp)
    if incoming_start is not None and incoming_start > entry_point.timestamp + 1.0:
        return "instrumental_entry"
    if "phrase" in exit_point.kinds and "phrase" in entry_point.kinds:
        return "phrase_crossfade"
    if exit_point.timestamp >= source.duration * 0.75 and entry_point.timestamp <= destination.duration * 0.2:
        return "outro_intro"
    source_level = _energy_at(source, exit_point.timestamp)
    destination_level = _energy_at(destination, entry_point.timestamp)
    if destination_level - source_level >= 0.2:
        return "energy_rise"
    if source_level - destination_level >= 0.2:
        return "energy_drop"
    return "instrumental_entry" if _instrumental_known(source, destination) else "phrase_crossfade"


def _instrumental_known(source: TrackAnalysis, destination: TrackAnalysis) -> bool:
    return source.structure.vocal_activity.available and destination.structure.vocal_activity.available


def _energy_at(track: TrackAnalysis, timestamp: float) -> float:
    if not track.energy.timeline:
        return track.energy.global_level
    return min(track.energy.timeline, key=lambda point: abs(point.timestamp - timestamp)).value


def _vocal_probability(estimate: VocalActivityEstimate, start: float, end: float) -> float:
    length = max(end - start, 1e-12)
    weighted = 0.0
    for segment in estimate.segments:
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        weighted += overlap * segment.probability
    return float(np.clip(weighted / length, 0.0, 1.0))


def _vocal_overlap(
    source: VocalActivityEstimate,
    destination: VocalActivityEstimate,
    source_time: float,
    destination_time: float,
    duration: float,
) -> float:
    """Mean product of aligned vocal probabilities across the overlap window."""
    total = 0.0
    for left in source.segments:
        left_start = max(0.0, left.start - source_time)
        left_end = min(duration, left.end - source_time)
        if left_end <= left_start:
            continue
        for right in destination.segments:
            right_start = max(0.0, right.start - destination_time)
            right_end = min(duration, right.end - destination_time)
            overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
            total += overlap * left.probability * right.probability
    return float(np.clip(total / max(duration, 1e-12), 0.0, 1.0))


def _incoming_vocal_start(destination: TrackAnalysis, entry: float) -> float | None:
    estimate = destination.structure.vocal_activity
    return estimate.first_significant_start_after(entry) if estimate.available else None


def _component(name: str, score: float, confidence: float, reason: str) -> TransitionComponent:
    return TransitionComponent(
        name=name,
        score=round(float(np.clip(score, 0.0, 1.0)), 3),
        confidence=round(float(np.clip(confidence, 0.0, 1.0)), 3),
        weight=TRANSITION_WEIGHTS[name],
        reason=reason,
    )
