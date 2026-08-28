"""Held-out comparison and diagnostic metrics for DJ-ranking experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ai_dj.models import TransitionQualityModel, evaluate_baseline, predict_transition_quality


def evaluate_ranking_systems(dataset_directory: str | Path, model: TransitionQualityModel) -> dict[str, Any]:
    """Compare baseline, ML, and hybrid predictions on the immutable test split."""
    test_path = Path(dataset_directory) / "test.jsonl"
    rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Held-out test split is empty; cannot compare ranking systems")
    baseline = np.asarray([float(row["features"]["transition_features"]["overall_score"]) for row in rows])
    ml = np.asarray([float(predict_transition_quality(model, row)["score"]) for row in rows])
    confidence = np.asarray([float(row["features"]["transition_features"].get("confidence", 0.0)) for row in rows])
    hybrid = np.clip(0.8 * ml + 0.2 * confidence, 0.0, 1.0)
    return {
        "row_count": len(rows),
        "baseline": _metrics(rows, baseline),
        "ml": _metrics(rows, ml),
        "hybrid": _metrics(rows, hybrid),
        "system_indicators": _system_indicators(rows),
        "failure_analysis": analyze_failures(rows, baseline, ml),
        "note": "Technical artifact rate is unavailable until audio rendering exists; it is not inferred from plans.",
    }


def analyze_failures(rows: list[dict[str, Any]], baseline: np.ndarray, ml: np.ndarray) -> dict[str, Any]:
    """Categorize where ML improves on, loses to, or ties the baseline target error."""
    labels = np.asarray([float(row["label"]) for row in rows])
    categories: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    for row, baseline_score, ml_score, label in zip(rows, baseline, ml, labels, strict=True):
        baseline_error = abs(baseline_score - label)
        ml_error = abs(ml_score - label)
        outcomes["ml_better" if ml_error < baseline_error else "baseline_better" if baseline_error < ml_error else "tie"] += 1
        categories[_failure_category(row)] += 1
    return {"outcomes": dict(sorted(outcomes.items())), "categories": dict(sorted(categories.items()))}


def _metrics(rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray([float(row["label"]) for row in rows])
    error = labels - predictions
    pearson = _correlation(labels, predictions)
    return {
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 6),
        "pearson_correlation": pearson,
        "spearman_correlation": _correlation(_ranks(labels), _ranks(predictions)),
        "pairwise_accuracy": _pairwise_accuracy(rows, labels, predictions),
    }


def _system_indicators(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    components = [row["features"]["transition_features"].get("components", {}) for row in rows]
    def mean_score(name: str) -> float | None:
        values = [float(item[name]["score"]) for item in components if name in item]
        return round(float(np.mean(values)), 6) if values else None
    vocal = mean_score("vocals")
    energy = mean_score("energy")
    return {
        "mean_beat_alignment": mean_score("beat_alignment"),
        "estimated_vocal_collision_rate": round(1.0 - vocal, 6) if vocal is not None else None,
        "mean_energy_discontinuity": round(1.0 - energy, 6) if energy is not None else None,
        "technical_artifact_rate": None,
    }


def _pairwise_accuracy(rows: list[dict[str, Any]], labels: np.ndarray, predictions: np.ndarray) -> float | None:
    correct = total = 0
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row.get("track_a_id", "all")), []).append(index)
    for indices in groups.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if labels[left] == labels[right]:
                    continue
                total += 1
                if (labels[left] - labels[right]) * (predictions[left] - predictions[right]) > 0:
                    correct += 1
    return round(correct / total, 6) if total else None


def _failure_category(row: dict[str, Any]) -> str:
    features = row["features"]
    components = features["transition_features"].get("components", {})
    for name, label in (("vocals", "vocal_collision"), ("phrase_alignment", "structure_error"), ("energy", "energy_error"), ("beat_alignment", "bpm_or_beat_error")):
        component = components.get(name)
        if component and float(component.get("confidence", 0.0)) >= 0.5 and float(component.get("score", 1.0)) < 0.45:
            return label
    harmonic = features.get("harmonic_features", {})
    if float(harmonic.get("confidence", 0.0)) >= 0.5 and float(harmonic.get("score", 1.0)) < 0.45:
        return "wrong_key_or_harmony"
    return "uncategorized"


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return round(float(np.corrcoef(left, right)[0, 1]), 6)
