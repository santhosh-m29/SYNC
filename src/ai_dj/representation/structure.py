"""Typed, confidence-scored representations for musical structure analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Bar:
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Bar":
        return cls(start=float(value["start"]), end=float(value["end"]), confidence=float(value["confidence"]))


@dataclass(frozen=True, slots=True)
class Phrase:
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Phrase":
        return cls(start=float(value["start"]), end=float(value["end"]), confidence=float(value["confidence"]))


@dataclass(frozen=True, slots=True)
class Section:
    start: float
    end: float
    label: str
    repetition_id: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "repetition_id": self.repetition_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Section":
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            label=str(value["label"]),
            repetition_id=str(value["repetition_id"]),
            confidence=float(value["confidence"]),
        )


@dataclass(frozen=True, slots=True)
class VocalActivity:
    start: float
    end: float
    probability: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "probability": self.probability}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VocalActivity":
        return cls(start=float(value["start"]), end=float(value["end"]), probability=float(value["probability"]))


@dataclass(frozen=True, slots=True)
class VocalActivityEstimate:
    segments: tuple[VocalActivity, ...]
    available: bool
    method: str
    confidence: float = 1.0

    @classmethod
    def unavailable(cls) -> "VocalActivityEstimate":
        return cls(segments=(), available=False, method="unavailable", confidence=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "available": self.available,
            "method": self.method,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VocalActivityEstimate":
        return cls(
            segments=tuple(VocalActivity.from_dict(item) for item in value["segments"]),
            available=bool(value["available"]),
            method=str(value["method"]),
            confidence=float(value.get("confidence", 0.0)),
        )

    def probability_at(self, timestamp: float) -> float:
        """Return the trusted piecewise vocal probability at ``timestamp``."""
        return max((segment.probability for segment in self.segments if segment.start <= timestamp < segment.end), default=0.0)

    def first_significant_start_after(self, timestamp: float, threshold: float = 0.5) -> float | None:
        """Find the first meaningful vocal entrance without inventing one."""
        starts = [max(timestamp, segment.start) for segment in self.segments if segment.end > timestamp and segment.probability >= threshold]
        return min(starts) if starts else None


@dataclass(frozen=True, slots=True)
class StructureAnalysis:
    bars: tuple[Bar, ...]
    phrases: tuple[Phrase, ...]
    sections: tuple[Section, ...]
    vocal_activity: VocalActivityEstimate

    @classmethod
    def empty(cls) -> "StructureAnalysis":
        return cls(bars=(), phrases=(), sections=(), vocal_activity=VocalActivityEstimate.unavailable())

    def to_dict(self) -> dict[str, Any]:
        return {
            "bars": [bar.to_dict() for bar in self.bars],
            "phrases": [phrase.to_dict() for phrase in self.phrases],
            "sections": [section.to_dict() for section in self.sections],
            "vocal_activity": self.vocal_activity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructureAnalysis":
        return cls(
            bars=tuple(Bar.from_dict(item) for item in value["bars"]),
            phrases=tuple(Phrase.from_dict(item) for item in value["phrases"]),
            sections=tuple(Section.from_dict(item) for item in value["sections"]),
            vocal_activity=VocalActivityEstimate.from_dict(value["vocal_activity"]),
        )
