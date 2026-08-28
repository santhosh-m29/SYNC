"""First learned transition-quality regressor and reproducible evaluation.

The model predicts an engineered transition-quality target; it does not
understand music or render audio. It consumes precomputed dataset features only.
"""

from __future__ import annotations

import json
import pickle
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MODEL_VERSION = "1.0"
MODEL_TYPE = "HistGradientBoostingRegressor"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_version: str = MODEL_VERSION
    seed: int = 7
    max_iter: int = 100
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 5


@dataclass(frozen=True, slots=True)
class TransitionQualityModel:
    estimator: Any
    feature_names: tuple[str, ...]
    model_version: str
    dataset_version: str
    feature_version: str
    analysis_versions: tuple[str, ...]


def train_transition_quality_model(
    dataset_directory: str | Path,
    artifact_directory: str | Path,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Any]:
    """Fit a simple tabular regressor using predefined dataset splits.

    The caller supplies a versioned dataset directory. This never reshuffles
    tracks or examples, and refuses to overwrite an existing model artifact.
    """
    dataset = _load_dataset(dataset_directory)
    _validate_dataset_for_training(dataset)
    baseline_metrics = evaluate_dataset_baseline(dataset_directory)
    feature_names = _feature_names(dataset["train"])
    train_x, train_y = _matrix_and_labels(dataset["train"], feature_names)
    from sklearn.ensemble import HistGradientBoostingRegressor

    estimator = HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.seed,
    )
    estimator.fit(train_x, train_y)
    metadata = _read_metadata(dataset_directory)
    model = TransitionQualityModel(
        estimator=estimator,
        feature_names=feature_names,
        model_version=config.model_version,
        dataset_version=str(metadata["dataset_version"]),
        feature_version=str(metadata["feature_version"]),
        analysis_versions=tuple(metadata["analysis_versions"]),
    )
    report = {
        "model_version": config.model_version,
        "model_type": MODEL_TYPE,
        "dataset_version": metadata["dataset_version"],
        "feature_version": metadata["feature_version"],
        "analysis_versions": metadata["analysis_versions"],
        "training_config": asdict(config),
        "feature_names": list(feature_names),
        "split_metrics": {
            split: _evaluation_metrics(_labels(rows), _predict_rows(model, rows))
            for split, rows in dataset.items()
            if rows
        },
        "baseline_metrics": baseline_metrics,
        "feature_importance": _permutation_importance(model, dataset["validation"] or dataset["test"]),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    test_metrics = report["split_metrics"]["test"]
    baseline_test = report["baseline_metrics"]["test"]
    report["comparison"] = {
        "test_mae_delta_vs_baseline": _delta(test_metrics["mae"], baseline_test["mae"]),
        "improves_baseline_mae": test_metrics["mae"] < baseline_test["mae"],
        "interpretation": "Automatic labels equal the deterministic plan score; a zero-error automatic-label baseline cannot be improved on that target. Human labels are required for a meaningful quality comparison.",
    }
    _write_artifact(artifact_directory, model, report)
    return report


def evaluate_baseline(rows: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """Evaluate deterministic plan-score predictions against dataset labels."""
    row_list = list(rows)
    predictions = np.asarray([_number(row["features"]["transition_features"]["overall_score"]) for row in row_list])
    return _evaluation_metrics(_labels(row_list), predictions)


def evaluate_dataset_baseline(dataset_directory: str | Path) -> dict[str, dict[str, float | None]]:
    """Evaluate the deterministic plan-score baseline on immutable dataset splits."""
    return {split: evaluate_baseline(rows) for split, rows in _load_dataset(dataset_directory).items()}


def load_transition_quality_model(artifact_directory: str | Path) -> TransitionQualityModel:
    """Load a trusted local model artifact produced by this package."""
    path = Path(artifact_directory) / "model.pkl"
    with path.open("rb") as input_file:
        model = pickle.load(input_file)
    if not isinstance(model, TransitionQualityModel):
        raise ValueError("Artifact is not a TransitionQualityModel")
    return model


def predict_transition_quality(model: TransitionQualityModel, example: dict[str, Any]) -> dict[str, float | str]:
    """Predict a bounded quality score using the exact training feature schema."""
    matrix = np.asarray([_feature_row(example, model.feature_names)], dtype=float)
    score = float(np.clip(model.estimator.predict(matrix)[0], 0.0, 1.0))
    return {"score": round(score, 6), "model_version": model.model_version}


def _load_dataset(directory: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return {split: _read_jsonl(root / f"{split}.jsonl") for split in ("train", "validation", "test")}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split is missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_metadata(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))


def _validate_dataset_for_training(dataset: dict[str, list[dict[str, Any]]]) -> None:
    if not dataset["train"]:
        raise ValueError("Training split is empty")
    if not dataset["test"]:
        raise ValueError("Held-out test split is empty; cannot evaluate generalization")


def _feature_names(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    names = sorted(_flatten_features(rows[0]["features"]).keys())
    if not names:
        raise ValueError("No usable features in training data")
    return tuple(names)


def _feature_row(row: dict[str, Any], feature_names: tuple[str, ...]) -> list[float]:
    flat = _flatten_features(row["features"])
    return [flat.get(name, 0.0) for name in feature_names]


def _flatten_features(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten numeric engineered features, excluding direct baseline target fields."""
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if name == "transition_features.overall_score" or name.endswith(".strategy"):
                continue
            flattened.update(_flatten_features(item, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = _number(value)
    return flattened


def _matrix_and_labels(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([_feature_row(row, feature_names) for row in rows], dtype=float), _labels(rows)


def _labels(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_number(row["label"]) for row in rows], dtype=float)


def _predict_rows(model: TransitionQualityModel, rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    matrix = np.asarray([_feature_row(row, model.feature_names) for row in rows], dtype=float)
    return np.clip(model.estimator.predict(matrix), 0.0, 1.0)


def _evaluation_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0:
        return {"mae": None, "rmse": None, "pearson_correlation": None}
    error = actual - predicted
    correlation = None
    if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = round(float(np.corrcoef(actual, predicted)[0, 1]), 6)
    return {
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 6),
        "pearson_correlation": correlation,
    }


def _permutation_importance(model: TransitionQualityModel, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    if len(rows) < 2:
        return []
    matrix, labels = _matrix_and_labels(rows, model.feature_names)
    baseline_mae = _evaluation_metrics(labels, np.clip(model.estimator.predict(matrix), 0.0, 1.0))["mae"]
    importances: list[dict[str, float]] = []
    for index, name in enumerate(model.feature_names):
        permuted = matrix.copy()
        permuted[:, index] = permuted[::-1, index]
        mae = _evaluation_metrics(labels, np.clip(model.estimator.predict(permuted), 0.0, 1.0))["mae"]
        importances.append({"feature": name, "mae_increase": round(float(mae - baseline_mae), 6)})
    return sorted(importances, key=lambda item: (-item["mae_increase"], item["feature"]))


def _write_artifact(directory: str | Path, model: TransitionQualityModel, report: dict[str, Any]) -> None:
    root = Path(directory)
    if root.exists():
        raise FileExistsError(f"Model artifact already exists: {root}")
    root.mkdir(parents=True)
    with (root / "model.pkl").open("wb") as output:
        pickle.dump(model, output)
    (root / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "MODEL_CARD.md").write_text(_model_card(report), encoding="utf-8")


def _model_card(report: dict[str, Any]) -> str:
    return f"""# Transition Quality Model {report['model_version']}

## Purpose

Predict an engineered transition-quality target from precomputed tabular audio-analysis features. This model does not understand music, select a complete set, or render audio.

## Data and split policy

Dataset version: `{report['dataset_version']}`. Feature version: `{report['feature_version']}`. Predefined leakage-safe splits were used without reshuffling. Labels may be automatic weak labels or imported human/consensus labels.

## Model

`{report['model_type']}` with configuration recorded in `metrics.json`.

## Evaluation

Test metrics: `{json.dumps(report['split_metrics']['test'])}`. Deterministic baseline metrics: `{json.dumps(report['baseline_metrics']['test'])}`.

## Feature importance

Permutation importance measures predictive association in this artifact; it does not establish musical causation.

## Limitations and intended use

Automatic labels are derived from deterministic planning scores, so they cannot demonstrate superiority over that same baseline. Use this model only for research on candidate-ranking features; validate on diverse, legally usable tracks with human-rated, artist/album-group-safe holdouts before any user-facing use.
"""


def _number(value: Any) -> float:
    result = float(value)
    return result if np.isfinite(result) else 0.0


def _delta(value: float | None, baseline: float | None) -> float | None:
    return round(value - baseline, 6) if value is not None and baseline is not None else None
