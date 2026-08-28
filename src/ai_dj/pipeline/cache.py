"""Versioned analysis cache keyed by source identity and modification state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ai_dj.representation.track import ANALYSIS_VERSION, TrackAnalysis


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    size: int
    modified_ns: int
    analysis_version: str

    @classmethod
    def from_path(cls, path: str | Path, analysis_version: str = ANALYSIS_VERSION) -> "FileFingerprint":
        source = Path(path).resolve()
        stat = source.stat()
        return cls(path=str(source), size=stat.st_size, modified_ns=stat.st_mtime_ns, analysis_version=analysis_version)

    def to_dict(self) -> dict[str, str | int]:
        return {"path": self.path, "size": self.size, "modified_ns": self.modified_ns,
                "analysis_version": self.analysis_version}


class AnalysisCache:
    def __init__(self, directory: str | Path, *, analysis_version: str = ANALYSIS_VERSION) -> None:
        self.directory = Path(directory)
        self.analysis_version = analysis_version

    def get(self, source: str | Path) -> TrackAnalysis | None:
        fingerprint = FileFingerprint.from_path(source, self.analysis_version)
        path = self._entry_path(fingerprint.path)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("fingerprint") != fingerprint.to_dict():
                return None
            return TrackAnalysis.from_dict(value["analysis"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, source: str | Path, analysis: TrackAnalysis) -> Path:
        fingerprint = FileFingerprint.from_path(source, self.analysis_version)
        if analysis.analysis_version != self.analysis_version:
            raise ValueError("Analysis version does not match cache version")
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self._entry_path(fingerprint.path)
        payload = {"fingerprint": fingerprint.to_dict(), "analysis": analysis.to_dict()}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination

    def _entry_path(self, resolved_path: str) -> Path:
        digest = hashlib.sha256(resolved_path.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"
