from __future__ import annotations

from dataclasses import replace

import pytest

from ai_dj.representation.track import EnergyEstimate, TempoEstimate
from ai_dj.set_planning import SetPlanningConfig, compare_greedy_and_sequence_aware, plan_set, plan_set_greedy
from tests.test_transition_planner import _track


def test_beam_set_plan_is_structured_deterministic_and_has_no_repeat_tracks():
    start, library = _library()
    config = SetPlanningConfig(target_track_count=4, beam_width=4, energy_trajectory="build")

    first = plan_set(start, library, config)
    second = plan_set(start, reversed(library), config)

    assert first == second
    assert first.complete
    assert len(first.track_ids) == 4
    assert first.energy_trajectory == "build"
    assert len(set(first.track_ids)) == len(first.track_ids)
    assert len(first.transitions) == 3
    assert first.to_dict()["tracks"] == list(first.track_ids)
    assert all(step.transition.source_track_id == first.track_ids[index] for index, step in enumerate(first.steps))
    assert all(0.0 <= step.incremental_score <= 1.0 for step in first.steps)
    levels = {track.track_id: track.energy.global_level for track in library}
    for index, step in enumerate(first.steps, start=1):
        target = 0.35 + 0.5 * index / 3
        assert step.energy_score == pytest.approx(1.0 - abs(levels[step.track_id] - target), abs=1e-6)


def test_context_penalizes_recent_artist_repetition_and_compares_greedy_search():
    start, library = _library()
    config = SetPlanningConfig(
        target_track_count=4,
        beam_width=4,
        artist_by_track_id={"start": "artist-a", "a": "artist-b", "b": "artist-b", "c": "artist-c"},
    )
    comparison = compare_greedy_and_sequence_aware(start, library, config)

    assert comparison.greedy.search_method == "greedy"
    assert comparison.sequence_aware.search_method == "beam"
    assert comparison.sequence_aware.complete
    assert any(step.artist_repeat_penalty == 1.0 for step in comparison.sequence_aware.steps)


def test_small_library_and_impossible_transitions_are_not_fabricated():
    start, _ = _library()
    alone = plan_set(start, [], SetPlanningConfig(target_track_count=1))
    impossible = replace(_track("impossible", duration=32.0), tempo=TempoEstimate(180.0, 0.9))
    blocked = plan_set(start, [impossible], SetPlanningConfig(target_track_count=2))

    assert alone.complete and alone.track_ids == ("start",) and alone.transitions == ()
    assert not blocked.complete
    assert blocked.track_ids == ("start",)
    assert blocked.notes


def test_config_validation_and_greedy_plan():
    start, library = _library()
    with pytest.raises(ValueError, match="at least one"):
        plan_set(start, library, SetPlanningConfig(target_track_count=0))
    greedy = plan_set_greedy(start, library, SetPlanningConfig(target_track_count=3))
    assert greedy.complete
    assert greedy.search_method == "greedy"


def _library():
    start = _with_energy(_track("start", duration=32.0), 0.35)
    return start, [_with_energy(_track("a", duration=32.0), 0.45), _with_energy(_track("b", duration=32.0), 0.65), _with_energy(_track("c", duration=32.0), 0.85)]


def _with_energy(track, level):
    return replace(track, energy=EnergyEstimate(level, track.energy.timeline))
