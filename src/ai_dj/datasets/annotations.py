"""Human-annotation interchange for weakly labeled transition examples."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ai_dj.datasets.models import DatasetExample, HumanAnnotation

RATING_FIELDS = ("overall_quality", "musical_coherence", "rhythm", "harmony", "energy", "vocal_interaction")


def export_annotation_template(examples: Iterable[DatasetExample], path: str | Path) -> Path:
    """Write editable JSONL evaluation forms; it intentionally contains no audio."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for example in examples:
            row = {
                "example_id": example.example_id,
                "track_a_id": example.track_a_id,
                "track_b_id": example.track_b_id,
                "source_exit": example.source_exit,
                "destination_entry": example.destination_entry,
                "transition_duration": example.transition_duration,
                "evaluator_id": "",
                **{field: None for field in RATING_FIELDS},
            }
            output.write(json.dumps(row, sort_keys=True) + "\n")
    return destination


def import_annotations(path: str | Path) -> tuple[HumanAnnotation, ...]:
    """Read completed JSONL forms and validate their required 1–5 ratings."""
    annotations: list[HumanAnnotation] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not value.get("evaluator_id"):
            continue
        try:
            ratings = {field: int(value[field]) for field in RATING_FIELDS}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid annotation at line {line_number}") from error
        if any(rating < 1 or rating > 5 for rating in ratings.values()):
            raise ValueError(f"Annotation ratings must be in 1–5 at line {line_number}")
        annotations.append(HumanAnnotation(example_id=str(value["example_id"]), evaluator_id=str(value["evaluator_id"]), **ratings))
    return tuple(annotations)


def merge_human_annotations(
    examples: Iterable[DatasetExample],
    annotations: Iterable[HumanAnnotation],
) -> tuple[DatasetExample, ...]:
    """Return relabeled copies; automatic rows are never written over in place."""
    grouped: dict[str, list[HumanAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.example_id].append(annotation)
    merged: list[DatasetExample] = []
    for example in examples:
        ratings = grouped.get(example.example_id, [])
        if not ratings:
            merged.append(example)
            continue
        label = sum(annotation.overall_quality for annotation in ratings) / (5.0 * len(ratings))
        merged.append(replace(example, label=round(label, 6), label_source="consensus" if len(ratings) > 1 else "human"))
    return tuple(merged)
