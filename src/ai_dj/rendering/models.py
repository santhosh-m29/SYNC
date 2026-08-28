"""Typed configuration and results for plan-driven audio rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RenderConfig:
    sample_rate: int = 22_050
    source_pre_roll_seconds: float = 8.0
    destination_post_roll_seconds: float = 8.0
    maximum_tempo_adjustment: float = 0.12
    maximum_gain_db: float = 6.0
    peak_ceiling: float = 0.98
    maximum_beat_alignment_seconds: float = 0.05


@dataclass(frozen=True, slots=True)
class RenderResult:
    output_path: Path
    sample_rate: int
    duration: float
    transition_start: float
    transition_duration: float
    strategy: str
    time_stretch_rate: float
    beat_alignment_seconds: float
    source_gain_db: float
    destination_gain_db: float
    peak_protection_db: float
    peak: float
    rms: float
    clipping_samples: int
