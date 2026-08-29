from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_dj.datasets import (
    DatasetConfig,
    build_dataset,
    export_annotation_template,
    generate_dataset,
    import_annotations,
    merge_human_annotations,
)
from ai_dj.representation.structure import Bar, Phrase, StructureAnalysis, VocalActivityEstimate
from ai_dj.representation.track import (
    BeatEstimate,
    DownbeatEstimate,
    EnergyEstimate,
    EnergyPoint,
    KeyEstimate,
    SpectralFeatures,
    TempoEstimate,
    TrackAnalysis,
)


def test_dataset_rows_have_versioned_schema_and_automatic_labels():
    result = build_dataset(_tracks(9), DatasetConfig(max_candidates_per_pair=3, split_ratios=(1 / 3, 1 / 3, 1 / 3)))
    examples = result.examples

    assert examples
    example = examples[0]
    row = example.to_dict()
    for field in (
        "track_a_id", "track_b_id", "source_exit", "destination_entry", "transition_duration",
        "bpm_a", "bpm_b", "tempo_difference", "harmonic_features", "rhythm_features",
        "energy_features", "structure_features", "vocal_features", "spectral_features",
        "transition_features", "label", "label_source", "dataset_version", "analysis_version", "feature_version",
    ):
        assert field in row
    assert example.label_source == "automatic"
    assert 0.0 <= example.label <= 1.0
    assert example.features["transition_features"]["strategy"]
    assert {"safety", "collision_duration", "maximum_overlap_probability", "integrated_overlap"} <= set(example.features["vocal_features"])


def test_dataset_generation_is_reproducible_and_stratifies_pair_candidates():
    config = DatasetConfig(max_candidates_per_pair=4, seed=99, split_ratios=(1 / 3, 1 / 3, 1 / 3))
    first = build_dataset(_tracks(9), config)
    second = build_dataset(_tracks(9), config)

    assert [example.to_dict() for example in first.examples] == [example.to_dict() for example in second.examples]
    assert all(len(examples) <= 4 * len(_split_track_ids(first, split)) * (len(_split_track_ids(first, split)) - 1) for split, examples in first.splits.items())
    assert {example.label for example in first.examples}


def test_split_assignments_prevent_track_and_group_leakage():
    tracks = _tracks(9)
    groups = {track.track_id: f"artist-{index // 3}" for index, track in enumerate(tracks)}
    result = build_dataset(_tracks(9), DatasetConfig(group_by_track_id=groups, split_ratios=(1 / 3, 1 / 3, 1 / 3)))
    assignments = result.metadata["track_split_assignments"]

    for index in range(0, 9, 3):
        assert len({assignments[track.track_id] for track in tracks[index : index + 3]}) == 1
    for split, examples in result.splits.items():
        assert all(assignments[example.track_a_id] == split == assignments[example.track_b_id] for example in examples)


def test_artifact_writer_creates_jsonl_splits_metadata_and_never_overwrites(tmp_path):
    destination = tmp_path / "datasets" / "v1"
    result = generate_dataset(_tracks(9), destination, DatasetConfig(max_candidates_per_pair=2, split_ratios=(1 / 3, 1 / 3, 1 / 3)))

    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"):
        assert (destination / name).is_file()
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["candidate_count"] == len(result.examples)
    with pytest.raises(FileExistsError):
        generate_dataset(_tracks(9), destination)


def test_annotation_export_import_and_human_consensus_do_not_mutate_automatic_examples(tmp_path):
    examples = build_dataset(_tracks(9), DatasetConfig(max_candidates_per_pair=1, split_ratios=(1 / 3, 1 / 3, 1 / 3))).examples
    template = export_annotation_template(examples[:1], tmp_path / "annotations.jsonl")
    row = json.loads(template.read_text(encoding="utf-8").strip())
    row.update({"evaluator_id": "listener-a", "overall_quality": 4, "musical_coherence": 4, "rhythm": 5, "harmony": 3, "energy": 4, "vocal_interaction": 2})
    second = {**row, "evaluator_id": "listener-b", "overall_quality": 2}
    template.write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n", encoding="utf-8")

    annotations = import_annotations(template)
    merged = merge_human_annotations(examples, annotations)

    assert examples[0].label_source == "automatic"
    relabeled = next(example for example in merged if example.example_id == annotations[0].example_id)
    assert relabeled.label_source == "consensus"
    assert relabeled.label == 0.6


def _split_track_ids(result, split: str) -> set[str]:
    return {track_id for track_id, assigned in result.metadata["track_split_assignments"].items() if assigned == split}


def _tracks(count: int) -> list[TrackAnalysis]:
    return [_track(f"track-{index}", bpm=112.0 + index * 3.0, key=("C major", "G major", "A minor")[index % 3]) for index in range(count)]


def _track(track_id: str, *, bpm: float, key: str) -> TrackAnalysis:
    duration = 32.0
    downbeats = tuple(float(value) for value in range(0, 32, 2))
    return TrackAnalysis(
        track_id=track_id,
        source_path=f"{track_id}.wav",
        duration=duration,
        tempo=TempoEstimate(bpm, 0.9),
        beats=BeatEstimate(tuple(float(value) / 2 for value in range(64)), 0.9),
        downbeats=DownbeatEstimate(downbeats, 0.9, 4),
        key=KeyEstimate(key, 0.9),
        energy=EnergyEstimate(0.7, (EnergyPoint(0.5, 0.7), EnergyPoint(31.5, 0.7))),
        spectral=SpectralFeatures(1000.0, 700.0, 2200.0, (10.0,), tuple(float(value) for value in range(13))),
        structure=StructureAnalysis(
            bars=tuple(Bar(start, start + 2.0, 0.9) for start in downbeats),
            phrases=tuple(Phrase(start, start + 8.0, 0.9) for start in (0.0, 8.0, 16.0, 24.0)),
            sections=(),
            vocal_activity=VocalActivityEstimate((), True, "trusted-silent-fixture"),
        ),
    )
