"""Typed records and configuration for transition-quality datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DATASET_VERSION = "1.0"
FEATURE_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_version: str = DATASET_VERSION
    feature_version: str = FEATURE_VERSION
    seed: int = 7
    max_candidates_per_pair: int = 24
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    provenance: str = "user-provided or licensed audio; provenance must be verified by the dataset owner"
    group_by_track_id: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "seed": self.seed,
            "max_candidates_per_pair": self.max_candidates_per_pair,
            "split_ratios": list(self.split_ratios),
            "provenance": self.provenance,
            "group_by_track_id": self.group_by_track_id or {},
            "candidate_strategy": "deterministic_stratified_transition_score",
            "label_strategy": "automatic_transition_plan_score",
        }


@dataclass(frozen=True, slots=True)
class DatasetExample:
    example_id: str
    track_a_id: str
    track_b_id: str
    source_exit: float
    destination_entry: float
    transition_duration: float
    strategy: str
    features: dict[str, Any]
    label: float
    label_source: str
    dataset_version: str
    analysis_version: str
    feature_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "track_a_id": self.track_a_id,
            "track_b_id": self.track_b_id,
            "source_exit": self.source_exit,
            "destination_entry": self.destination_entry,
            "transition_duration": self.transition_duration,
            "strategy": self.strategy,
            **self.features,
            "features": self.features,
            "label": self.label,
            "label_source": self.label_source,
            "dataset_version": self.dataset_version,
            "analysis_version": self.analysis_version,
            "feature_version": self.feature_version,
        }


@dataclass(frozen=True, slots=True)
class HumanAnnotation:
    example_id: str
    evaluator_id: str
    overall_quality: int
    musical_coherence: int
    rhythm: int
    harmony: int
    energy: int
    vocal_interaction: int


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    splits: dict[str, tuple[DatasetExample, ...]]
    metadata: dict[str, Any]

    @property
    def examples(self) -> tuple[DatasetExample, ...]:
        return tuple(example for split in ("train", "validation", "test") for example in self.splits[split])
