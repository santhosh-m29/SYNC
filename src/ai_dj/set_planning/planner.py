"""Context-aware deterministic DJ-set planning without audio rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from ai_dj.evaluation import assess_transition_constraints
from ai_dj.representation.track import TrackAnalysis
from ai_dj.set_planning.models import PlanningComparison, SetPlan, SetPlanningConfig, SetStep
from ai_dj.transition import find_best_transitions
from ai_dj.transition.models import TransitionPlan

# The score is deliberately inspectable. Pairwise transition quality remains
# dominant; energy direction and diversity guide selection without overpowering
# a musically strong transition.
OBJECTIVE_WEIGHTS = {"transition": 0.65, "energy": 0.20, "variety": 0.15}
_TRAJECTORY_ENDPOINTS = {
    "build": (0.35, 0.85),
    "maintain": (0.60, 0.60),
    "release": (0.85, 0.35),
    "peak": (0.85, 0.85),
}


@dataclass(frozen=True, slots=True)
class _SearchState:
    tracks: tuple[TrackAnalysis, ...]
    steps: tuple[SetStep, ...]
    total_score: float


def plan_set(
    start_track: TrackAnalysis,
    candidate_tracks: Iterable[TrackAnalysis],
    config: SetPlanningConfig = SetPlanningConfig(),
) -> SetPlan:
    """Use bounded beam search to find a context-aware, non-rendering set plan."""
    _validate_config(config)
    candidates = _unique_candidates(start_track, candidate_tracks)
    initial = _SearchState((start_track,), (), 0.0)
    beam = [initial]
    for _ in range(config.target_track_count - 1):
        expanded = [child for state in beam for child in _expand(state, candidates, config)]
        if not expanded:
            break
        beam = sorted(expanded, key=_state_key)[: config.beam_width]
    chosen = sorted(beam, key=_state_key)[0]
    return _to_plan(chosen, config, "beam")


def plan_set_greedy(
    start_track: TrackAnalysis,
    candidate_tracks: Iterable[TrackAnalysis],
    config: SetPlanningConfig = SetPlanningConfig(),
) -> SetPlan:
    """Choose the strongest immediate context-aware extension at each step."""
    _validate_config(config)
    candidates = _unique_candidates(start_track, candidate_tracks)
    state = _SearchState((start_track,), (), 0.0)
    for _ in range(config.target_track_count - 1):
        options = _expand(state, candidates, config)
        if not options:
            break
        state = sorted(options, key=_state_key)[0]
    return _to_plan(state, config, "greedy")


def compare_greedy_and_sequence_aware(
    start_track: TrackAnalysis,
    candidate_tracks: Iterable[TrackAnalysis],
    config: SetPlanningConfig = SetPlanningConfig(),
) -> PlanningComparison:
    """Evaluate greedy and beam search with the same candidates and objective."""
    # Materialize once so generators provide identical input to both searches.
    candidates = tuple(candidate_tracks)
    return PlanningComparison(
        greedy=plan_set_greedy(start_track, candidates, config),
        sequence_aware=plan_set(start_track, candidates, config),
    )


def _expand(state: _SearchState, candidates: tuple[TrackAnalysis, ...], config: SetPlanningConfig) -> list[_SearchState]:
    results: list[_SearchState] = []
    for destination in candidates:
        if not config.allow_track_repeats and destination.track_id in {track.track_id for track in state.tracks}:
            continue
        plan = _best_eligible_transition(state.tracks[-1], destination)
        if plan is None:
            continue
        step = _score_step(state.tracks, destination, plan, config)
        results.append(_SearchState(state.tracks + (destination,), state.steps + (step,), state.total_score + step.incremental_score))
    return results


def _best_eligible_transition(source: TrackAnalysis, destination: TrackAnalysis) -> TransitionPlan | None:
    for plan in find_best_transitions(source, destination):
        if assess_transition_constraints(source, destination, plan).allowed:
            return plan
    return None


def _score_step(history: tuple[TrackAnalysis, ...], destination: TrackAnalysis, plan: TransitionPlan, config: SetPlanningConfig) -> SetStep:
    progress = len(history) / max(config.target_track_count - 1, 1)
    target_energy = _target_energy(config.energy_trajectory, progress)
    energy_score = float(np.clip(1.0 - abs(destination.energy.global_level - target_energy), 0.0, 1.0))
    artist_penalty = _artist_repeat_penalty(history, destination, config)
    similarity_penalty = _similarity_penalty(history[-config.recent_history_size :], destination)
    variety_score = float(np.clip(1.0 - 0.6 * artist_penalty - 0.4 * similarity_penalty, 0.0, 1.0))
    incremental = (
        OBJECTIVE_WEIGHTS["transition"] * plan.overall_score
        + OBJECTIVE_WEIGHTS["energy"] * energy_score
        + OBJECTIVE_WEIGHTS["variety"] * variety_score
    )
    return SetStep(
        track_id=destination.track_id,
        transition=plan,
        transition_score=plan.overall_score,
        energy_score=round(energy_score, 6),
        variety_score=round(variety_score, 6),
        artist_repeat_penalty=round(artist_penalty, 6),
        similarity_penalty=round(similarity_penalty, 6),
        incremental_score=round(float(incremental), 6),
    )


def _target_energy(trajectory: str, progress: float) -> float:
    start, end = _TRAJECTORY_ENDPOINTS[trajectory]
    return start + (end - start) * float(np.clip(progress, 0.0, 1.0))


def _artist_repeat_penalty(history: tuple[TrackAnalysis, ...], destination: TrackAnalysis, config: SetPlanningConfig) -> float:
    if not config.artist_by_track_id:
        return 0.0
    artist = config.artist_by_track_id.get(destination.track_id)
    if artist is None:
        return 0.0
    return 1.0 if any(config.artist_by_track_id.get(track.track_id) == artist for track in history) else 0.0


def _similarity_penalty(history: tuple[TrackAnalysis, ...], destination: TrackAnalysis) -> float:
    if not history:
        return 0.0
    similarities = []
    for track in history:
        centroid_scale = max(track.spectral.centroid_hz, destination.spectral.centroid_hz, 1.0)
        timbre = max(0.0, 1.0 - abs(track.spectral.centroid_hz - destination.spectral.centroid_hz) / centroid_scale)
        tempo = max(0.0, 1.0 - abs(track.tempo.bpm - destination.tempo.bpm) / max(track.tempo.bpm, destination.tempo.bpm, 1.0))
        harmony = 1.0 if track.key.key is not None and track.key.key == destination.key.key else 0.0
        similarities.append((timbre + tempo + harmony) / 3.0)
    return float(np.clip(max(similarities), 0.0, 1.0))


def _state_key(state: _SearchState) -> tuple[float, float, tuple[str, ...]]:
    # Prefer reaching the requested length; among equal-length sequences use the
    # inspectable cumulative objective and stable IDs for deterministic ties.
    return (-len(state.tracks), -state.total_score, tuple(track.track_id for track in state.tracks))


def _to_plan(state: _SearchState, config: SetPlanningConfig, method: Literal["beam", "greedy"]) -> SetPlan:
    complete = len(state.tracks) == config.target_track_count
    notes = () if complete else ("No further eligible transition was found; this plan was not padded with invented tracks.",)
    score = state.total_score / len(state.steps) if state.steps else 0.0
    return SetPlan(
        track_ids=tuple(track.track_id for track in state.tracks),
        steps=state.steps,
        overall_score=round(score, 6),
        complete=complete,
        energy_trajectory=config.energy_trajectory,
        search_method=method,
        notes=notes,
    )


def _unique_candidates(start_track: TrackAnalysis, candidates: Iterable[TrackAnalysis]) -> tuple[TrackAnalysis, ...]:
    unique = {track.track_id: track for track in candidates if track.track_id != start_track.track_id}
    return tuple(unique[track_id] for track_id in sorted(unique))


def _validate_config(config: SetPlanningConfig) -> None:
    if config.target_track_count < 1:
        raise ValueError("target_track_count must be at least one")
    if config.beam_width < 1:
        raise ValueError("beam_width must be positive")
    if config.recent_history_size < 1:
        raise ValueError("recent_history_size must be positive")
    if config.energy_trajectory not in _TRAJECTORY_ENDPOINTS:
        raise ValueError(f"Unknown energy trajectory: {config.energy_trajectory}")
