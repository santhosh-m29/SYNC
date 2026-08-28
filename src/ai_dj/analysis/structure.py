"""Deterministic, bar-aligned musical structure candidates."""

from __future__ import annotations

import numpy as np

from ai_dj.analysis.vocals import estimate_vocal_activity
from ai_dj.representation.structure import Bar, Phrase, Section, StructureAnalysis
from ai_dj.representation.track import DownbeatEstimate, EnergyEstimate

PHRASE_BARS = 4
MIN_BARS_PER_SECTION = 2
REPETITION_SIMILARITY = 0.88


def analyze_structure(
    samples: np.ndarray,
    sample_rate: int,
    duration: float,
    downbeats: DownbeatEstimate,
    energy: EnergyEstimate,
) -> StructureAnalysis:
    """Generate bar, phrase, and non-semantic section candidates.

    Boundaries are derived from bar-aligned changes in chroma, MFCC, and the
    existing energy timeline. Repetition IDs represent self-similarity; labels
    remain ``other`` because semantic section names need stronger evidence.
    """
    vocal_activity = estimate_vocal_activity(samples, sample_rate)
    bars = infer_bars(duration, downbeats)
    if not bars:
        return StructureAnalysis(bars=(), phrases=(), sections=(), vocal_activity=vocal_activity)

    phrases = infer_phrases(bars)
    features = _bar_features(samples, sample_rate, bars, energy)
    boundaries, boundary_confidences = _boundary_indices(features)
    sections = _sections_from_boundaries(bars, duration, features, boundaries, boundary_confidences)
    return StructureAnalysis(bars=bars, phrases=phrases, sections=sections, vocal_activity=vocal_activity)


def infer_bars(duration: float, downbeats: DownbeatEstimate) -> tuple[Bar, ...]:
    """Convert inferred downbeats into contiguous bar candidates."""
    starts = sorted({float(timestamp) for timestamp in downbeats.timestamps if 0.0 <= timestamp < duration})
    if not starts:
        return ()
    confidence = float(np.clip(downbeats.confidence, 0.0, 1.0))
    bars = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else duration
        if end > start:
            bars.append(Bar(start=round(start, 6), end=round(end, 6), confidence=round(confidence, 3)))
    return tuple(bars)


def infer_phrases(bars: tuple[Bar, ...]) -> tuple[Phrase, ...]:
    """Group bar candidates into conventional four-bar phrase candidates.

    The grouping is applied to inferred bars rather than a fixed time signature;
    a partial final group receives reduced confidence.
    """
    phrases: list[Phrase] = []
    for start_index in range(0, len(bars), PHRASE_BARS):
        group = bars[start_index : start_index + PHRASE_BARS]
        if not group:
            continue
        completeness = len(group) / PHRASE_BARS
        confidence = float(np.mean([bar.confidence for bar in group]) * completeness)
        phrases.append(
            Phrase(start=group[0].start, end=group[-1].end, confidence=round(float(np.clip(confidence, 0.0, 1.0)), 3))
        )
    return tuple(phrases)


def _bar_features(
    samples: np.ndarray,
    sample_rate: int,
    bars: tuple[Bar, ...],
    energy: EnergyEstimate,
) -> np.ndarray:
    import librosa

    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=samples, sr=sample_rate, hop_length=hop_length, tuning=0.0)
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, hop_length=hop_length, n_mfcc=13)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate, hop_length=hop_length)
    rows: list[np.ndarray] = []
    for bar in bars:
        mask = (times >= bar.start) & (times < bar.end)
        if not np.any(mask):
            mask = np.array([int(np.argmin(np.abs(times - bar.start)))])
        harmonic = np.mean(chroma[:, mask], axis=1)
        timbral = np.mean(mfcc[:, mask], axis=1)
        level = _mean_energy(energy, bar.start, bar.end)
        rows.append(np.concatenate((harmonic, timbral, np.array([level], dtype=float))))
    matrix = np.vstack(rows)
    # Feature-wise normalization prevents MFCC-0 scale from dominating chroma,
    # timbre, and energy changes in the similarity calculation.
    centered = matrix - np.mean(matrix, axis=0)
    scale = np.std(centered, axis=0)
    normalized = centered / np.where(scale > 1e-8, scale, 1.0)
    return np.vstack([_unit_vector(row) for row in normalized])


def _mean_energy(energy: EnergyEstimate, start: float, end: float) -> float:
    values = [point.value for point in energy.timeline if start <= point.timestamp < end]
    return float(np.mean(values)) if values else energy.global_level


def _boundary_indices(features: np.ndarray) -> tuple[tuple[int, ...], dict[int, float]]:
    if len(features) < MIN_BARS_PER_SECTION * 2:
        return (), {}
    similarity = np.sum(features[:-1] * features[1:], axis=1)
    scores = np.clip((1.0 - similarity) / 2.0, 0.0, 1.0)
    threshold = float(np.median(scores) + 0.5 * np.std(scores))
    candidates: list[int] = []
    confidences: dict[int, float] = {}
    for position, score in enumerate(scores, start=1):
        if position < MIN_BARS_PER_SECTION or len(features) - position < MIN_BARS_PER_SECTION:
            continue
        left = scores[position - 2] if position > 1 else -np.inf
        right = scores[position] if position < len(scores) else -np.inf
        if score >= threshold and score >= left and score >= right:
            candidates.append(position)
            confidences[position] = float(np.clip(score, 0.0, 1.0))
    return tuple(candidates), confidences


def _sections_from_boundaries(
    bars: tuple[Bar, ...],
    duration: float,
    features: np.ndarray,
    boundaries: tuple[int, ...],
    boundary_confidences: dict[int, float],
) -> tuple[Section, ...]:
    indices = (0, *boundaries, len(bars))
    groups: list[np.ndarray] = []
    sections: list[Section] = []
    for section_index, (start_index, end_index) in enumerate(zip(indices, indices[1:]), start=1):
        vector = _unit_vector(np.mean(features[start_index:end_index], axis=0))
        repetition_id = _repetition_id(vector, groups)
        if repetition_id == len(groups) + 1:
            groups.append(vector)
        start = 0.0 if start_index == 0 else bars[start_index].start
        end = duration if end_index == len(bars) else bars[end_index].start
        left_confidence = boundary_confidences.get(start_index, bars[start_index].confidence)
        right_confidence = boundary_confidences.get(end_index, bars[end_index - 1].confidence)
        confidence = float(np.clip((left_confidence + right_confidence) / 2.0, 0.0, 1.0))
        sections.append(
            Section(
                start=round(start, 6),
                end=round(end, 6),
                label="other",
                repetition_id=f"section_{repetition_id}",
                confidence=round(confidence, 3),
            )
        )
    return tuple(sections)


def _repetition_id(vector: np.ndarray, groups: list[np.ndarray]) -> int:
    if not groups:
        return 1
    similarities = [float(np.dot(vector, candidate)) for candidate in groups]
    best_index = int(np.argmax(similarities))
    return best_index + 1 if similarities[best_index] >= REPETITION_SIMILARITY else len(groups) + 1


def _unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-12 else np.zeros_like(values)
