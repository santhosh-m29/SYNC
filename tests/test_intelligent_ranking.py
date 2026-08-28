from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ai_dj.evaluation import (
    assess_transition_constraints,
    evaluate_ranking_systems,
    export_blind_comparisons,
    rank_next_tracks_ml,
    rank_transition_candidates,
    summarize_blind_preferences,
)
from ai_dj.models import TrainingConfig, load_transition_quality_model, train_transition_quality_model
from ai_dj.representation.track import TempoEstimate
from ai_dj.transition import find_best_transition
from tests.test_transition_planner import _track
from tests.test_transition_quality_model import _write_fixture_dataset


def test_ml_baseline_and_hybrid_rankings_are_separate_and_constrained(tmp_path):
    model = _fixture_model(tmp_path)
    current = _track("current", duration=32.0)
    compatible = _track("compatible", duration=32.0)
    excessive_tempo = replace(_track("too-fast", duration=32.0), tempo=TempoEstimate(150.0, 0.9))

    comparison = rank_transition_candidates(current, [compatible, excessive_tempo], model)

    assert comparison.baseline and comparison.ml and comparison.hybrid
    assert all(item.plan.destination_track_id == "compatible" for item in comparison.ml)
    assert comparison.hybrid[0].final_score == round(
        0.8 * comparison.hybrid[0].ml_score + 0.2 * comparison.hybrid[0].plan.confidence, 6
    )
    assert rank_next_tracks_ml(current, [compatible], model) == comparison.ml
    rejected = find_best_transition(current, excessive_tempo)
    assert rejected is not None
    constraint = assess_transition_constraints(current, excessive_tempo, rejected)
    assert not constraint.allowed
    assert "12%" in constraint.reasons[0]


def test_constraints_reject_invalid_timestamps_and_missing_tempo():
    source = _track("source", duration=32.0)
    destination = _track("destination", duration=32.0)
    plan = find_best_transition(source, destination)

    assert plan is not None
    invalid_timestamp = replace(plan, source_exit=source.duration)
    assert not assess_transition_constraints(source, destination, invalid_timestamp).allowed
    missing_tempo = replace(destination, tempo=TempoEstimate(0.0, 0.0))
    result = assess_transition_constraints(source, missing_tempo, plan)
    assert not result.allowed
    assert result.tempo_adjustment is None


def test_held_out_comparison_and_blind_manifest(tmp_path):
    dataset = _write_fixture_dataset(tmp_path / "dataset")
    model = _fixture_model(tmp_path, dataset)
    report = evaluate_ranking_systems(dataset, model)

    assert report["row_count"] == 12
    assert report["baseline"]["mae"] is not None
    assert report["ml"]["spearman_correlation"] is not None
    assert report["hybrid"]["pairwise_accuracy"] is not None
    assert report["system_indicators"]["technical_artifact_rate"] is None
    assert "outcomes" in report["failure_analysis"]

    current = _track("current", duration=32.0)
    target = _track("target", duration=32.0)
    rankings = rank_transition_candidates(current, [target], model)
    output = tmp_path / "blind"
    export_blind_comparisons([( "comparison-1", rankings.baseline[0], rankings.ml[0], rankings.hybrid[0])], output, seed=3)
    public = (output / "blind_evaluation.jsonl").read_text(encoding="utf-8")
    assert "baseline" not in public and "hybrid" not in public
    key = json.loads((output / "blind_answer_key.jsonl").read_text(encoding="utf-8"))
    choice = next(option for option, system in key["option_systems"].items() if system == "ml")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"comparison_id": "comparison-1", "preferred_option": choice}) + "\n", encoding="utf-8")
    assert summarize_blind_preferences(annotations, output / "blind_answer_key.jsonl") == {"baseline": 0, "ml": 1, "hybrid": 0}


def test_evaluation_refuses_an_empty_held_out_split(tmp_path):
    dataset = _write_fixture_dataset(tmp_path / "dataset")
    model = _fixture_model(tmp_path, dataset)
    (dataset / "test.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Held-out test split is empty"):
        evaluate_ranking_systems(dataset, model)


def _fixture_model(tmp_path, dataset=None):
    dataset = dataset or _write_fixture_dataset(tmp_path / "dataset")
    artifact = tmp_path / "model"
    if not artifact.exists():
        train_transition_quality_model(dataset, artifact, TrainingConfig(seed=5, max_iter=20, min_samples_leaf=2))
    return load_transition_quality_model(artifact)
