"""Turn analyzed candidates into future events on the source media clock."""
from dataclasses import dataclass, replace
import math

from ai_dj.matching import score_track_pair
from ai_dj.transition import find_best_transitions
from ai_dj.transition.models import TransitionPlan
from ai_dj.transition.vocal_safety import assess_vocal_safety, tempo_stretch_rate, VocalSafetyConfig


@dataclass(frozen=True)
class PlaybackEvent:
    generation: int
    current_track: str
    next_track: str
    outgoing_transition_timestamp: float
    incoming_start_timestamp: float
    transition_duration: float  # wall-clock seconds
    tempo_ratio: float          # incoming media seconds per output second
    beat_offset: float
    strategy: str
    confidence: float
    vocal_safety: str
    reason: str
    transition_score: float = 0.0


def plan_event(source, destination, position=0.0, source_rate=1.0, generation=0,
               cue=None, transition=None, duration=None, maximum_tempo_adjustment=0.08):
    effective = replace(source, tempo=replace(source.tempo, bpm=source.tempo.bpm * source_rate))
    ratio = tempo_stretch_rate(effective.tempo.bpm, destination.tempo.bpm)
    tempo_ok = (min(source.tempo.confidence, destination.tempo.confidence) >= 0.25
                and abs(ratio - 1) <= maximum_tempo_adjustment)
    rate = ratio if tempo_ok else 1.0
    lead = min(2 * source_rate, max(0.0, (source.duration - position) / 4))
    candidates = find_best_transitions(effective, destination, include_rejected=True, search_durations=True)
    # Bound search cost and avoid starting an automatic cue halfway through lyrics.
    events = []
    for plan in candidates:
        exit_time = transition if transition is not None else plan.source_exit
        entry = cue if cue is not None else plan.destination_entry
        offset = 0.0
        if cue is None and min(source.beats.confidence, destination.beats.confidence) >= .5:
            if source.beats.timestamps and destination.beats.timestamps:
                out_beat = min(source.beats.timestamps, key=lambda t: abs(t - exit_time))
                in_beat = min(destination.beats.timestamps, key=lambda t: abs(t - entry))
                offset = max(-.05, min(.05, (out_beat - exit_time) / source_rate - (in_beat - entry) / rate))
                aligned = max(0.0, entry - offset * rate)
                offset = (entry - aligned) / rate
                entry = aligned
        if exit_time < position + lead and transition is None:
            continue
        if transition is not None and exit_time < position:
            continue  # manual point remains stored, but is no longer a future event
        vocal = destination.structure.vocal_activity
        if cue is None and vocal.available and any(
            s.probability >= .6 and s.start + .05 < entry < s.end for s in vocal.segments
        ):
            continue
        # Durations are integer bars where meter/tempo evidence supports this.
        beat_seconds = 60 / effective.tempo.bpm if tempo_ok else 0
        requested = duration if duration is not None else plan.duration / source_rate
        if duration is None and beat_seconds and source.downbeats.confidence >= .3:
            bar = beat_seconds * (source.downbeats.meter or 4)
            requested = max(bar, math.floor(requested / bar) * bar)
        if not tempo_ok and duration is None:
            requested = min(requested, .25)
        # Limit uncertain overlap, but still search for likely instrumental windows.
        evidence = min(source.structure.vocal_activity.confidence, vocal.confidence)
        if evidence < .6 and duration is None:
            requested = min(requested, 2.0)
        seconds = min(requested, (source.duration - exit_time) / source_rate,
                      (destination.duration - entry) / rate)
        if seconds < .02 or entry < 0:
            continue
        mapped_source = replace(source, structure=replace(source.structure,
            vocal_activity=replace(source.structure.vocal_activity, segments=tuple(
                replace(s, start=s.start / source_rate, end=s.end / source_rate)
                for s in source.structure.vocal_activity.segments))))
        probe = replace(plan, source_exit=exit_time / source_rate, destination_entry=entry, duration=seconds)
        safety = assess_vocal_safety(mapped_source, destination, probe,
                                    VocalSafetyConfig(minimum_significant_seconds=0.0),
                                    destination_rate=rate, destination_offset=0.0)
        # Uncertain evidence can rank windows; it cannot certify safety.
        collision = _collision(source, destination, exit_time, entry, seconds, source_rate, rate)
        if collision * seconds > .25 and duration is None:
            continue  # even uncertain positive vocal evidence should avoid sustained overlap
        state = "safe" if safety.allowed else "collision" if safety.verified else "uncertain" if (
            source.structure.vocal_activity.available and vocal.available) else "unavailable"
        if state == "collision" and duration is None:
            continue
        score = plan.overall_score - collision * .8
        events.append((score, PlaybackEvent(generation, source.track_id, destination.track_id,
            exit_time, entry, seconds, rate, offset,
            "beat_crossfade" if tempo_ok else "tempo_unmatched_short_fade",
            min(plan.confidence, evidence), state, safety.reason, plan.overall_score)))
    if events:
        return max(events, key=lambda item: (item[0], item[1].outgoing_transition_timestamp))[1]
    # Keep sound continuous with a very short de-clicked handoff. Never call it vocal-safe.
    entry = cue if cue is not None else 0.0
    seconds = min(duration if duration is not None else .05,
                  (source.duration - position) / source_rate / 2,
                  (destination.duration - entry) / rate)
    exit_time = transition if transition is not None and transition >= position else source.duration - seconds * source_rate
    seconds = min(seconds, (source.duration - exit_time) / source_rate)
    if seconds <= 0 or not 0 <= entry < destination.duration:
        raise ValueError("No remaining audio for the requested transition")
    return PlaybackEvent(generation, source.track_id, destination.track_id, exit_time, entry,
                         seconds, rate, 0.0, "short_handoff", 0.0, "unverified",
                         "No verified future overlap; short smooth handoff", 0.0)


def _collision(source, destination, exit_time, entry, duration, source_rate, rate):
    left, right = source.structure.vocal_activity, destination.structure.vocal_activity
    if not left.available or not right.available:
        return 0.0
    collision = 0.0
    for a in left.segments:
        if a.end <= exit_time or a.start >= exit_time + duration * source_rate:
            continue
        for b in right.segments:
            if b.end <= entry or b.start >= entry + duration * rate:
                continue
            start = max(0, (a.start - exit_time) / source_rate, (b.start - entry) / rate)
            end = min(duration, (a.end - exit_time) / source_rate, (b.end - entry) / rate)
            collision += max(0, end - start) * a.probability * b.probability
    return min(1.0, collision / duration)


def choose_next(source, candidates):
    """Choose a destination using its best *actual* exit/entry combination."""
    ranked = []
    # Bound expensive local-window search for large libraries.
    candidates = sorted(candidates, key=lambda t: score_track_pair(source, t).overall_score, reverse=True)[:8]
    for candidate in candidates:
        pair = score_track_pair(source, candidate)
        plans = find_best_transitions(source, candidate)
        if not plans:
            continue
        best = plans[0]
        # Pair compatibility gets context from the selected musical window;
        # a strong global pair cannot rescue a bad set of local candidates.
        score = 0.65 * best.overall_score + 0.35 * pair.overall_score
        ranked.append((score, best.confidence, candidate.track_id, candidate))
    if not ranked:
        raise ValueError(f"No compatible transition candidate from {source.track_id}")
    return max(ranked, key=lambda item: item[:3])[-1]
