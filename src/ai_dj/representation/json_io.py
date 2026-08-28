"""Reliable JSON persistence for track analyses."""

from __future__ import annotations

import json
from pathlib import Path

from ai_dj.representation.track import TrackAnalysis


def write_analysis(path: str | Path, analysis: TrackAnalysis) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(analysis.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_analysis(path: str | Path) -> TrackAnalysis:
    return TrackAnalysis.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
