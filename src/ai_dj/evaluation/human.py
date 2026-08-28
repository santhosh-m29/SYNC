"""Blind-plan comparison manifests for future human listening evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from ai_dj.evaluation.ranking import RankedTransition


def export_blind_comparisons(
    comparisons: Iterable[tuple[str, RankedTransition, RankedTransition, RankedTransition]], output_directory: str | Path, seed: int = 7
) -> None:
    """Write evaluator-safe manifests and a separate answer key without rendering audio."""
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"Blind evaluation output already exists: {root}")
    root.mkdir(parents=True)
    public_rows, key_rows = [], []
    rng = random.Random(seed)
    for comparison_id, baseline, ml, hybrid in comparisons:
        choices = [("baseline", baseline), ("ml", ml), ("hybrid", hybrid)]
        rng.shuffle(choices)
        option_ids = ("option_a", "option_b", "option_c")
        public_rows.append({"comparison_id": comparison_id, "options": [{"option_id": option, "source_track_id": item.plan.source_track_id, "destination_track_id": item.plan.destination_track_id, "source_exit": item.plan.source_exit, "destination_entry": item.plan.destination_entry, "duration": item.plan.duration} for option, (_, item) in zip(option_ids, choices, strict=True)], "ratings": {"overall_quality": "1-5", "musical_coherence": "1-5", "rhythm": "1-5", "harmony": "1-5", "energy": "1-5", "vocal_interaction": "1-5"}})
        key_rows.append({"comparison_id": comparison_id, "option_systems": {option: system for option, (system, _) in zip(option_ids, choices, strict=True)}})
    _write_jsonl(root / "blind_evaluation.jsonl", public_rows)
    _write_jsonl(root / "blind_answer_key.jsonl", key_rows)


def summarize_blind_preferences(annotation_path: str | Path, answer_key_path: str | Path) -> dict[str, int]:
    """Map imported preferred options back to systems; annotations stay separate from labels."""
    keys = {row["comparison_id"]: row["option_systems"] for row in _read_jsonl(answer_key_path)}
    counts = {"baseline": 0, "ml": 0, "hybrid": 0}
    for row in _read_jsonl(annotation_path):
        system = keys.get(row.get("comparison_id"), {}).get(row.get("preferred_option"))
        if system in counts:
            counts[system] += 1
    return counts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
