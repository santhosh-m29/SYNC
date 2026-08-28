from __future__ import annotations

import json

import pytest

from ai_dj.models import (
    TrainingConfig,
    evaluate_baseline,
    load_transition_quality_model,
    predict_transition_quality,
    train_transition_quality_model,
)


def test_training_loading_inference_and_reproducible_artifact_metrics(tmp_path):
    dataset = _write_fixture_dataset(tmp_path / "dataset")
    first_artifact = tmp_path / "models" / "v1"
    config = TrainingConfig(seed=13, max_iter=30, min_samples_leaf=2)

    first_report = train_transition_quality_model(dataset, first_artifact, config)
    model = load_transition_quality_model(first_artifact)
    test_row = _read_rows(dataset / "test.jsonl")[0]
    prediction = predict_transition_quality(model, test_row)

    assert (first_artifact / "model.pkl").is_file()
    assert (first_artifact / "metrics.json").is_file()
    assert (first_artifact / "MODEL_CARD.md").is_file()
    assert 0.0 <= prediction["score"] <= 1.0
    assert prediction["model_version"] == "1.0"
    assert first_report["split_metrics"]["test"]["mae"] is not None
    assert first_report["baseline_metrics"]["test"]["mae"] is not None
    assert first_report["feature_importance"]
    assert "does not understand music" in (first_artifact / "MODEL_CARD.md").read_text(encoding="utf-8")

    second_report = train_transition_quality_model(dataset, tmp_path / "models" / "v2", config)
    assert first_report["split_metrics"] == second_report["split_metrics"]
    with pytest.raises(FileExistsError):
        train_transition_quality_model(dataset, first_artifact, config)


def test_baseline_uses_deterministic_transition_score_and_empty_test_is_rejected(tmp_path):
    dataset = _write_fixture_dataset(tmp_path / "dataset")
    test_rows = _read_rows(dataset / "test.jsonl")
    metrics = evaluate_baseline(test_rows)

    assert metrics["mae"] > 0.0  # fixture labels represent independent human-like ratings.
    (dataset / "test.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Held-out test split is empty"):
        train_transition_quality_model(dataset, tmp_path / "models" / "empty", TrainingConfig(min_samples_leaf=2))


def _write_fixture_dataset(root):
    root.mkdir(parents=True)
    for split, offset, count in (("train", 0, 36), ("validation", 36, 12), ("test", 48, 12)):
        rows = [_row(index) for index in range(offset, offset + count)]
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = {"dataset_version": "fixture-1", "feature_version": "1.0", "analysis_versions": ["2.0"]}
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _row(index: int):
    quality_signal = (index % 12) / 11
    # The target is intentionally not the automatic baseline; it mimics an
    # independently rated fixture so model-vs-baseline metrics are meaningful.
    label = 0.15 + 0.7 * quality_signal + (0.05 if index % 2 else -0.03)
    baseline = 1.0 - quality_signal
    features = {
        "bpm_a": 110 + index % 6,
        "bpm_b": 112 + index % 6,
        "tempo_difference": 2.0,
        "harmonic_features": {"score": quality_signal, "confidence": 0.9},
        "rhythm_features": {"pair": {"score": quality_signal, "confidence": 0.9}, "candidate": {"score": quality_signal, "confidence": 0.9}},
        "energy_features": {"source_global": quality_signal, "destination_global": quality_signal, "difference": 0.0, "candidate": {"score": quality_signal, "confidence": 1.0}},
        "structure_features": {"source_phrase_count": 4, "destination_phrase_count": 4, "pair": {"score": quality_signal, "confidence": 0.9}, "candidate": {"score": quality_signal, "confidence": 0.9}, "phrase_alignment": {"score": quality_signal, "confidence": 0.9}},
        "vocal_features": {"score": quality_signal, "confidence": 0.0},
        "spectral_features": {"pair": {"score": quality_signal, "confidence": 1.0}, "source_centroid_hz": 900.0, "destination_centroid_hz": 1000.0},
        "transition_features": {"strategy": "phrase_crossfade", "overall_score": baseline, "confidence": 0.9, "components": {"tempo": {"score": quality_signal, "confidence": 0.9}}},
    }
    return {"example_id": f"example-{index}", "features": features, "label": label, "label_source": "human"}
