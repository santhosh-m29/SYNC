"""ML-assisted transition ranking with explicit technical guardrails.

This module plans no audio and does not replace deterministic planning.  It
keeps baseline, learned, and hybrid rankings separate so they can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ai_dj.datasets import example_from_transition_plan
from ai_dj.models import TransitionQualityModel, predict_transition_quality
from ai_dj.representation.track import TrackAnalysis
from ai_dj.transition import find_best_transitions
from ai_dj.transition.models import TransitionPlan
from ai_dj.transition.vocal_safety import assess_vocal_safety

RankingSystem = Literal["baseline", "ml", "hybrid"]
_MAX_TEMPO_ADJUSTMENT = 0.12


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Whether a plan is technically eligible for ranking."""

    allowed: bool
    reasons: tuple[str, ...]
    tempo_adjustment: float | None


@dataclass(frozen=True, slots=True)
class RankedTransition:
    """One selected plan and its scores under a named ranking system."""

    system: RankingSystem
    plan: TransitionPlan
    final_score: float
    deterministic_score: float
    ml_score: float | None
    confidence_adjustment: float
    constraints: ConstraintResult
    explanation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankingComparison:
    """The same candidate library ordered by each independent system."""

    baseline: tuple[RankedTransition, ...]
    ml: tuple[RankedTransition, ...]
    hybrid: tuple[RankedTransition, ...]


def assess_transition_constraints(
    source: TrackAnalysis, destination: TrackAnalysis, plan: TransitionPlan
) -> ConstraintResult:
    """Apply non-negotiable validity and practical tempo constraints.

    A ratio near 1:1, 1:2, or 2:1 is acceptable.  The residual correction must
    stay within 12%; this is a conservative planning threshold, not rendering.
    """
    reasons: list[str] = []
    if plan.source_track_id != source.track_id or plan.destination_track_id != destination.track_id:
        reasons.append("Plan track identifiers do not match its analyzed tracks")
    if source.duration <= 0.0 or destination.duration <= 0.0:
        reasons.append("A track has no usable duration")
    if not (0.0 <= plan.source_exit < source.duration and 0.0 <= plan.destination_entry < destination.duration):
        reasons.append("Plan timestamps are outside track bounds")
    if plan.duration < 0.0 or plan.source_exit + plan.duration > source.duration or plan.destination_entry + plan.duration > destination.duration:
        reasons.append("Plan duration exceeds available audio")
    vocal_safety = assess_vocal_safety(source, destination, plan)
    if vocal_safety.verified and not vocal_safety.allowed:
        reasons.append(vocal_safety.reason)
    if source.tempo.bpm <= 0.0 or destination.tempo.bpm <= 0.0 or source.tempo.confidence <= 0.0 or destination.tempo.confidence <= 0.0:
        reasons.append("Required tempo analysis is unavailable")
        return ConstraintResult(False, tuple(reasons), None)

    ratio = source.tempo.bpm / destination.tempo.bpm
    adjustment = min(abs(ratio / relationship - 1.0) for relationship in (0.5, 1.0, 2.0))
    if adjustment > _MAX_TEMPO_ADJUSTMENT:
        reasons.append(f"Tempo correction {adjustment:.1%} exceeds the 12% planning limit")
    return ConstraintResult(not reasons, tuple(reasons), round(adjustment, 6))


def rank_transition_candidates(
    current_track: TrackAnalysis,
    candidate_tracks: Sequence[TrackAnalysis],
    model: TransitionQualityModel,
) -> RankingComparison:
    """Plan each pair and return baseline, ML, and hybrid rankings separately."""
    scored: dict[RankingSystem, list[RankedTransition]] = {"baseline": [], "ml": [], "hybrid": []}
    for candidate in candidate_tracks:
        plans = find_best_transitions(current_track, candidate)
        scored["baseline"].extend(
            _ranked_for_plans("baseline", current_track, candidate, plans, model)
        )
        scored["ml"].extend(_ranked_for_plans("ml", current_track, candidate, plans, model))
        scored["hybrid"].extend(_ranked_for_plans("hybrid", current_track, candidate, plans, model))
    return RankingComparison(
        baseline=_best_per_destination(scored["baseline"]),
        ml=_best_per_destination(scored["ml"]),
        hybrid=_best_per_destination(scored["hybrid"]),
    )


def rank_next_tracks_ml(
    current_track: TrackAnalysis, candidate_tracks: Sequence[TrackAnalysis], model: TransitionQualityModel
) -> tuple[RankedTransition, ...]:
    """Return eligible candidates ordered by learned transition-quality score."""
    return rank_transition_candidates(current_track, candidate_tracks, model).ml


def _ranked_for_plans(
    system: RankingSystem,
    source: TrackAnalysis,
    destination: TrackAnalysis,
    plans: Sequence[TransitionPlan],
    model: TransitionQualityModel,
) -> list[RankedTransition]:
    results: list[RankedTransition] = []
    for plan in plans:
        constraints = assess_transition_constraints(source, destination, plan)
        if not constraints.allowed:
            continue
        deterministic = plan.overall_score
        ml_score = float(predict_transition_quality(model, example_from_transition_plan(source, destination, plan).to_dict())["score"])
        confidence_adjustment = 0.2 * plan.confidence
        final = deterministic if system == "baseline" else ml_score if system == "ml" else 0.8 * ml_score + confidence_adjustment
        explanation = _explanation(system, deterministic, ml_score, plan, constraints)
        results.append(
            RankedTransition(
                system=system,
                plan=plan,
                final_score=round(final, 6),
                deterministic_score=deterministic,
                ml_score=round(ml_score, 6),
                confidence_adjustment=round(confidence_adjustment if system == "hybrid" else 0.0, 6),
                constraints=constraints,
                explanation=explanation,
            )
        )
    return results


def _best_per_destination(results: Sequence[RankedTransition]) -> tuple[RankedTransition, ...]:
    by_track: dict[str, RankedTransition] = {}
    for result in results:
        track_id = result.plan.destination_track_id
        prior = by_track.get(track_id)
        if prior is None or _sort_key(result) < _sort_key(prior):
            by_track[track_id] = result
    return tuple(sorted(by_track.values(), key=_sort_key))


def _sort_key(result: RankedTransition) -> tuple[float, float, str]:
    return (-result.final_score, -result.plan.confidence, result.plan.destination_track_id)


def _explanation(
    system: RankingSystem, deterministic: float, ml_score: float, plan: TransitionPlan, constraints: ConstraintResult
) -> tuple[str, ...]:
    if system == "baseline":
        return ("Ranked by the deterministic transition-planning score.",)
    if system == "ml":
        return ("Ranked by the learned transition-quality prediction.",)
    return (
        "Eligible under hard timestamp, audio-duration, and tempo constraints.",
        f"Hybrid score = 0.8 × ML ({ml_score:.3f}) + 0.2 × plan confidence ({plan.confidence:.3f}).",
        f"Deterministic score remains available for comparison ({deterministic:.3f}).",
    )
