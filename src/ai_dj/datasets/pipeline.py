"""Reproducible generation of weakly labeled transition datasets."""

from __future__ import annotations

import hashlib
import json
import math
import random
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

from ai_dj.datasets.models import DatasetBuildResult, DatasetConfig, DatasetExample
from ai_dj.matching import score_track_pair
from ai_dj.representation.track import TrackAnalysis
from ai_dj.transition import find_best_transitions
from ai_dj.transition.models import TransitionPlan


def build_dataset(tracks: Iterable[TrackAnalysis], config: DatasetConfig = DatasetConfig()) -> DatasetBuildResult:
    """Build deterministic, weakly labeled examples without writing an artifact.

    Track/group assignment occurs before pair generation, so no track or caller-
    supplied artist/album group can appear in multiple splits. Cross-split pairs
    are intentionally excluded to prevent direct track leakage.
    """
    track_list = sorted(tracks, key=lambda track: track.track_id)
    _validate_config(config)
    if len({track.analysis_version for track in track_list}) > 1:
        raise ValueError("All tracks in a dataset must use the same analysis_version")
    assignments = _assign_splits(track_list, config)
    split_tracks = {
        split: [track for track in track_list if assignments[track.track_id] == split]
        for split in ("train", "validation", "test")
    }
    splits = {
        split: tuple(
            example
            for source, destination in permutations(split_tracks[split], 2)
            for example in _examples_for_pair(source, destination, config)
        )
        for split in ("train", "validation", "test")
    }
    metadata = _metadata(track_list, splits, assignments, config)
    return DatasetBuildResult(splits=splits, metadata=metadata)


def generate_dataset(
    tracks: Iterable[TrackAnalysis],
    output_directory: str | Path,
    config: DatasetConfig = DatasetConfig(),
) -> DatasetBuildResult:
    """Build and write a new versioned dataset directory without overwriting it."""
    result = build_dataset(tracks, config)
    _write_dataset(output_directory, result)
    return result


def _examples_for_pair(source: TrackAnalysis, destination: TrackAnalysis, config: DatasetConfig) -> tuple[DatasetExample, ...]:
    plans = _stratified_plans(find_best_transitions(source, destination), config.max_candidates_per_pair)
    pair = score_track_pair(source, destination)
    return tuple(_example_from_plan(source, destination, pair, plan, config) for plan in plans)


def example_from_transition_plan(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    plan: TransitionPlan,
    config: DatasetConfig = DatasetConfig(),
) -> DatasetExample:
    """Create the canonical model feature row for one already-planned transition.

    Online inference uses this same transformation as dataset generation.  The
    automatic label is retained solely because ``DatasetExample`` is a
    versioned row type; callers must not interpret it as a new quality rating.
    """
    return _example_from_plan(source, destination, score_track_pair(source, destination), plan, config)


def _stratified_plans(plans: list[TransitionPlan], limit: int) -> tuple[TransitionPlan, ...]:
    """Retain score-stratified candidates to include weak and strong examples."""
    if len(plans) <= limit:
        return tuple(plans)
    ordered = sorted(plans, key=lambda plan: (plan.overall_score, plan.source_exit, plan.destination_entry, plan.duration))
    indices = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)} if limit > 1 else {0}
    return tuple(ordered[index] for index in sorted(indices))


def _example_from_plan(
    source: TrackAnalysis,
    destination: TrackAnalysis,
    pair,
    plan: TransitionPlan,
    config: DatasetConfig,
) -> DatasetExample:
    transition = {component.name: {"score": component.score, "confidence": component.confidence} for component in plan.components}
    compatibility = {component.name: {"score": component.score, "confidence": component.confidence} for component in pair.components}
    features: dict[str, Any] = {
        "bpm_a": source.tempo.bpm,
        "bpm_b": destination.tempo.bpm,
        "tempo_difference": abs(source.tempo.bpm - destination.tempo.bpm),
        "harmonic_features": compatibility["harmony"],
        "rhythm_features": {"pair": compatibility["rhythm"], "candidate": transition["beat_alignment"]},
        "energy_features": {
            "source_global": source.energy.global_level,
            "destination_global": destination.energy.global_level,
            "difference": abs(source.energy.global_level - destination.energy.global_level),
            "candidate": transition["energy"],
        },
        "structure_features": {
            "source_phrase_count": len(source.structure.phrases),
            "destination_phrase_count": len(destination.structure.phrases),
            "pair": compatibility["structure"],
            "candidate": transition["structure"],
            "phrase_alignment": transition["phrase_alignment"],
        },
        "vocal_features": transition["vocals"],
        "spectral_features": {
            "pair": compatibility["timbre"],
            "source_centroid_hz": source.spectral.centroid_hz,
            "destination_centroid_hz": destination.spectral.centroid_hz,
        },
        "transition_features": {
            "strategy": plan.strategy,
            "overall_score": plan.overall_score,
            "confidence": plan.confidence,
            "components": transition,
        },
    }
    example_id = _example_id(source.track_id, destination.track_id, plan)
    return DatasetExample(
        example_id=example_id,
        track_a_id=source.track_id,
        track_b_id=destination.track_id,
        source_exit=plan.source_exit,
        destination_entry=plan.destination_entry,
        transition_duration=plan.duration,
        strategy=plan.strategy,
        features=features,
        label=plan.overall_score,
        label_source="automatic",
        dataset_version=config.dataset_version,
        analysis_version=source.analysis_version,
        feature_version=config.feature_version,
    )


def _assign_splits(tracks: list[TrackAnalysis], config: DatasetConfig) -> dict[str, str]:
    groups = {track.track_id: (config.group_by_track_id or {}).get(track.track_id, track.track_id) for track in tracks}
    unique_groups = sorted(set(groups.values()))
    shuffled = unique_groups[:]
    random.Random(config.seed).shuffle(shuffled)
    train_count, validation_count = _split_counts(len(shuffled), config.split_ratios)
    group_splits = {
        **{group: "train" for group in shuffled[:train_count]},
        **{group: "validation" for group in shuffled[train_count : train_count + validation_count]},
        **{group: "test" for group in shuffled[train_count + validation_count :]},
    }
    return {track_id: group_splits[group] for track_id, group in groups.items()}


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    if total == 0:
        return 0, 0
    train = int(math.floor(total * ratios[0]))
    validation = int(math.floor(total * ratios[1]))
    if total >= 3:
        train = max(train, 1)
        validation = max(validation, 1)
        if train + validation >= total:
            train = max(1, total - 2)
            validation = 1
    return train, validation


def _metadata(
    tracks: list[TrackAnalysis],
    splits: dict[str, tuple[DatasetExample, ...]],
    assignments: dict[str, str],
    config: DatasetConfig,
) -> dict[str, Any]:
    labels = [example.label for examples in splits.values() for example in examples]
    return {
        **config.to_dict(),
        "analysis_versions": sorted({track.analysis_version for track in tracks}),
        "track_count": len(tracks),
        "track_split_assignments": assignments,
        "split_counts": {split: len(examples) for split, examples in splits.items()},
        "candidate_count": len(labels),
        "label_distribution": {
            "low_[0,.33)": sum(label < 0.33 for label in labels),
            "medium_[.33,.66)": sum(0.33 <= label < 0.66 for label in labels),
            "high_[.66,1]": sum(label >= 0.66 for label in labels),
        },
    }


def _write_dataset(output_directory: str | Path, result: DatasetBuildResult) -> None:
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"Dataset output already exists: {root}")
    root.mkdir(parents=True)
    for split, examples in result.splits.items():
        _write_jsonl(root / f"{split}.jsonl", (example.to_dict() for example in examples))
    (root / "metadata.json").write_text(json.dumps(result.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _example_id(source_id: str, destination_id: str, plan: TransitionPlan) -> str:
    value = f"{source_id}|{destination_id}|{plan.source_exit:.6f}|{plan.destination_entry:.6f}|{plan.duration:.3f}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _validate_config(config: DatasetConfig) -> None:
    if config.max_candidates_per_pair < 1:
        raise ValueError("max_candidates_per_pair must be positive")
    if len(config.split_ratios) != 3 or any(value < 0 for value in config.split_ratios):
        raise ValueError("split_ratios must contain three non-negative values")
    if not math.isclose(sum(config.split_ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split_ratios must sum to 1.0")
