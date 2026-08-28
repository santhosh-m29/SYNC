"""Versioned, extensible representation of one analyzed track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_dj.representation.structure import StructureAnalysis

ANALYSIS_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class TempoEstimate:
    bpm: float
    confidence: float
    octave_alternatives: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"bpm": self.bpm, "confidence": self.confidence}
        if self.octave_alternatives:
            value["octave_alternatives"] = list(self.octave_alternatives)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TempoEstimate":
        return cls(
            bpm=float(value["bpm"]),
            confidence=float(value["confidence"]),
            octave_alternatives=tuple(float(item) for item in value.get("octave_alternatives", ())),
        )


@dataclass(frozen=True, slots=True)
class BeatEstimate:
    timestamps: tuple[float, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"timestamps": list(self.timestamps), "confidence": self.confidence}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BeatEstimate":
        return cls(timestamps=tuple(float(item) for item in value["timestamps"]), confidence=float(value["confidence"]))


@dataclass(frozen=True, slots=True)
class DownbeatEstimate:
    timestamps: tuple[float, ...]
    confidence: float
    meter: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"timestamps": list(self.timestamps), "confidence": self.confidence, "meter": self.meter}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DownbeatEstimate":
        meter = value.get("meter")
        return cls(
            timestamps=tuple(float(item) for item in value["timestamps"]),
            confidence=float(value["confidence"]),
            meter=int(meter) if meter is not None else None,
        )


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    key: str | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KeyEstimate":
        key = value.get("key")
        return cls(key=str(key) if key is not None else None, confidence=float(value["confidence"]))


@dataclass(frozen=True, slots=True)
class EnergyPoint:
    timestamp: float
    value: float

    def to_dict(self) -> dict[str, float]:
        return {"timestamp": self.timestamp, "value": self.value}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EnergyPoint":
        return cls(timestamp=float(value["timestamp"]), value=float(value["value"]))


@dataclass(frozen=True, slots=True)
class EnergyEstimate:
    global_level: float
    timeline: tuple[EnergyPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"global": self.global_level, "timeline": [point.to_dict() for point in self.timeline]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EnergyEstimate":
        return cls(
            global_level=float(value["global"]),
            timeline=tuple(EnergyPoint.from_dict(item) for item in value["timeline"]),
        )


@dataclass(frozen=True, slots=True)
class SpectralFeatures:
    centroid_hz: float
    bandwidth_hz: float
    rolloff_hz: float
    contrast: tuple[float, ...]
    mfcc: tuple[float, ...]

    @classmethod
    def zero(cls) -> "SpectralFeatures":
        return cls(centroid_hz=0.0, bandwidth_hz=0.0, rolloff_hz=0.0, contrast=(), mfcc=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroid_hz": self.centroid_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "rolloff_hz": self.rolloff_hz,
            "contrast": list(self.contrast),
            "mfcc": list(self.mfcc),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpectralFeatures":
        return cls(
            centroid_hz=float(value["centroid_hz"]),
            bandwidth_hz=float(value["bandwidth_hz"]),
            rolloff_hz=float(value["rolloff_hz"]),
            contrast=tuple(float(item) for item in value["contrast"]),
            mfcc=tuple(float(item) for item in value["mfcc"]),
        )


@dataclass(frozen=True, slots=True)
class TrackAnalysis:
    track_id: str
    source_path: str
    duration: float
    tempo: TempoEstimate
    beats: BeatEstimate
    downbeats: DownbeatEstimate = DownbeatEstimate(timestamps=(), confidence=0.0, meter=None)
    key: KeyEstimate = KeyEstimate(key=None, confidence=0.0)
    energy: EnergyEstimate = EnergyEstimate(global_level=0.0, timeline=())
    spectral: SpectralFeatures = SpectralFeatures.zero()
    structure: StructureAnalysis = StructureAnalysis.empty()
    analysis_version: str = ANALYSIS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "source_path": self.source_path,
            "duration": self.duration,
            "tempo": self.tempo.to_dict(),
            "beats": self.beats.to_dict(),
            "downbeats": self.downbeats.to_dict(),
            "key": self.key.to_dict(),
            "energy": self.energy.to_dict(),
            "spectral": self.spectral.to_dict(),
            "structure": self.structure.to_dict(),
            "analysis_version": self.analysis_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrackAnalysis":
        return cls(
            track_id=str(value["track_id"]),
            source_path=str(value["source_path"]),
            duration=float(value["duration"]),
            tempo=TempoEstimate.from_dict(value["tempo"]),
            beats=BeatEstimate.from_dict(value["beats"]),
            downbeats=DownbeatEstimate.from_dict(value["downbeats"])
            if "downbeats" in value
            else DownbeatEstimate(timestamps=(), confidence=0.0, meter=None),
            key=KeyEstimate.from_dict(value["key"]) if "key" in value else KeyEstimate(key=None, confidence=0.0),
            energy=EnergyEstimate.from_dict(value["energy"])
            if "energy" in value
            else EnergyEstimate(global_level=0.0, timeline=()),
            spectral=SpectralFeatures.from_dict(value["spectral"])
            if "spectral" in value
            else SpectralFeatures.zero(),
            structure=StructureAnalysis.from_dict(value["structure"])
            if "structure" in value
            else StructureAnalysis.empty(),
            analysis_version=str(value["analysis_version"]),
        )

    @classmethod
    def track_id_for(cls, source: str | Path) -> str:
        import hashlib

        return hashlib.sha256(str(Path(source).resolve()).encode("utf-8")).hexdigest()[:20]
