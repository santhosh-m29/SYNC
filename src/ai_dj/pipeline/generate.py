"""End-to-end orchestration from a local library to a rendered WAV set.

Candidate retrieval, set planning, and rendering remain separate modules.  This
layer coordinates them and records which deterministic fallbacks were used.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from ai_dj.pipeline.analyze import LibraryAnalysisResult, analyze_library
from ai_dj.pipeline.cache import AnalysisCache
from ai_dj.rendering import RenderConfig, RenderResult, render_transition, separate_stems
from ai_dj.representation.json_io import write_analysis
from ai_dj.representation.track import TrackAnalysis
from ai_dj.set_planning import SetPlan, SetPlanningConfig, plan_set
from ai_dj.transition import find_best_transitions


class GenerationError(RuntimeError):
    """Raised when the library cannot produce a safely renderable set."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    target_track_count: int = 4
    energy_trajectory: Literal["build", "maintain", "release", "peak"] = "maintain"
    candidate_pool_size: int = 24
    beam_width: int = 8
    seed: int = 7
    start_track_id: str | None = None
    sample_rate: int = 22_050
    set_join_seconds: float = 0.02
    use_vocal_stems: bool = False
    stem_cache_directory: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    output_path: Path
    report_path: Path
    set_plan: SetPlan
    render_results: tuple[RenderResult, ...]
    metrics: dict[str, Any]
    analysis_failures: tuple[str, ...]
    fallbacks: tuple[str, ...]


def generate_dj_set(
    input_directory: str | Path,
    output_path: str | Path,
    config: GenerationConfig = GenerationConfig(),
    *,
    cache_directory: str | Path | None = None,
    analysis_output_directory: str | Path | None = None,
) -> GenerationResult:
    """Analyze a library, plan an eligible set, render it, and write a report.

    This version intentionally uses the deterministic planner. A valid learned
    artifact is optional future input; the local automatic-label dataset cannot
    yet establish a model suitable for autonomous use.
    """
    started = time.perf_counter()
    root = Path(input_directory).expanduser()
    _validate_config(config)
    analysis_started = time.perf_counter()
    analysis_result = analyze_library(root, AnalysisCache(cache_directory or root / ".ai_dj_cache"))
    analysis_seconds = time.perf_counter() - analysis_started
    if analysis_output_directory is not None:
        for analysis in analysis_result.analyses:
            write_analysis(Path(analysis_output_directory) / f"{analysis.track_id}.json", analysis)
    if len(analysis_result.analyses) < 2:
        raise GenerationError("At least two successfully analyzed tracks are required")

    planning_started = time.perf_counter()
    start = _choose_start_track(analysis_result.analyses, config)
    retrieved = retrieve_candidates(start, analysis_result.analyses, config.candidate_pool_size)
    planning_config = SetPlanningConfig(
        target_track_count=min(config.target_track_count, len(retrieved) + 1),
        beam_width=config.beam_width,
        energy_trajectory=config.energy_trajectory,
    )
    set_plan = plan_set(start, retrieved, planning_config)
    planning_seconds = time.perf_counter() - planning_started
    if not set_plan.steps:
        raise GenerationError("No eligible transition plan could be generated from this library")

    rendering_started = time.perf_counter()
    output = Path(output_path)
    segment_directory = output.parent / f".{output.stem}_segments"
    segment_directory.mkdir(parents=True, exist_ok=True)
    tracks = {track.track_id: track for track in analysis_result.analyses}
    if config.use_vocal_stems:
        set_plan = _upgrade_to_stem_crossfades(set_plan, tracks)
    renders: list[RenderResult] = []
    for index, step in enumerate(set_plan.steps, start=1):
        source = tracks[step.transition.source_track_id]
        destination = tracks[step.transition.destination_track_id]
        source_stems = destination_stems = None
        if config.use_vocal_stems:
            stem_root = Path(config.stem_cache_directory) if config.stem_cache_directory else root / ".ai_dj_stems"
            source_stems = separate_stems(source.source_path, stem_root)
            destination_stems = separate_stems(destination.source_path, stem_root)
        result = render_transition(
            source,
            destination,
            step.transition,
            segment_directory / f"{index:02d}_{source.track_id}_{destination.track_id}.wav",
            RenderConfig(sample_rate=config.sample_rate),
            source_stems=source_stems,
            destination_stems=destination_stems,
        )
        _validate_render(result)
        renders.append(result)
    joins = _assemble_set(tuple(renders), output, config.sample_rate, config.set_join_seconds)
    rendering_seconds = time.perf_counter() - rendering_started
    quality = _set_quality(output, renders, joins)
    _validate_set_quality(quality)
    fallbacks = ["deterministic_transition_and_set_scoring"]
    if config.use_vocal_stems:
        fallbacks.append("demucs_stem_vocal_handoff")
    if any(track.tempo.confidence < 0.25 for track in tracks.values() if track.track_id in set_plan.track_ids):
        fallbacks.append("low_tempo_confidence_used_existing_technical_constraints")
    metrics = {
        "tracks_selected": len(set_plan.track_ids),
        "transitions_generated": len(renders),
        "average_transition_score": round(float(np.mean([step.transition_score for step in set_plan.steps])), 6),
        "lowest_transition_score": round(float(min(step.transition_score for step in set_plan.steps)), 6),
        "bpm_range": [round(min(tracks[item].tempo.bpm for item in set_plan.track_ids), 6), round(max(tracks[item].tempo.bpm for item in set_plan.track_ids), 6)],
        "energy_trajectory": config.energy_trajectory,
        "technical_failures": 0,
        "fallbacks_used": fallbacks,
        "quality": quality,
        "timing_seconds": {
            "analysis": round(analysis_seconds, 6),
            "candidate_retrieval_and_planning": round(planning_seconds, 6),
            "rendering": round(rendering_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "candidate_retrieval": {"pool_size": len(retrieved), "method": "tempo_key_energy_prefilter"},
        "configuration": asdict(config),
    }
    report = {
        "set_plan": set_plan.to_dict(),
        "tracks": [{"track_id": track_id, "source_path": tracks[track_id].source_path} for track_id in set_plan.track_ids],
        "renders": [asdict(item) | {"output_path": str(item.output_path)} for item in renders],
        "metrics": metrics,
        "analysis": {"successful": len(analysis_result.analyses), "failed": len(analysis_result.failures), "cached": analysis_result.cached, "skipped": analysis_result.skipped},
        "limitations": ["ML ranking was not used because no held-out validated model artifact is available.", "The final WAV is an ordered assembly of validated transition previews; it is not a full-length mastered DJ mix."],
    }
    report_path = output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_human_report(output.with_suffix(".report.txt"), set_plan, tracks, metrics, fallbacks)
    return GenerationResult(output, report_path, set_plan, tuple(renders), metrics, tuple(str(path) for path, _ in analysis_result.failures), tuple(fallbacks))


def retrieve_candidates(current: TrackAnalysis, tracks: list[TrackAnalysis], limit: int) -> tuple[TrackAnalysis, ...]:
    """Cheap deterministic prefilter before expensive transition candidate search."""
    if limit < 1:
        raise ValueError("candidate_pool_size must be positive")
    candidates = [track for track in tracks if track.track_id != current.track_id]
    def key(track: TrackAnalysis) -> tuple[float, str]:
        tempo = abs(current.tempo.bpm - track.tempo.bpm) / max(current.tempo.bpm, track.tempo.bpm, 1.0)
        energy = abs(current.energy.global_level - track.energy.global_level)
        key_penalty = 0.0 if current.key.key is None or track.key.key is None or current.key.key == track.key.key else 0.2
        return tempo + 0.5 * energy + key_penalty, track.track_id
    return tuple(sorted(candidates, key=key)[:limit])


def _upgrade_to_stem_crossfades(set_plan: SetPlan, tracks: dict[str, TrackAnalysis]) -> SetPlan:
    """Restore smooth prior crossfades only when renderer-owned stems are ready."""
    previous_id = set_plan.track_ids[0]
    updated_steps = []
    for step in set_plan.steps:
        plans = find_best_transitions(tracks[previous_id], tracks[step.track_id], include_rejected=True)
        smooth = next((plan for plan in plans if plan.duration > 0.0), step.transition)
        updated_steps.append(replace(step, transition=smooth, transition_score=smooth.overall_score))
        previous_id = step.track_id
    return replace(set_plan, steps=tuple(updated_steps), notes=set_plan.notes + ("Stem-backed vocal handoffs restore smooth accompaniment crossfades.",))


def _choose_start_track(tracks: list[TrackAnalysis], config: GenerationConfig) -> TrackAnalysis:
    if config.start_track_id is not None:
        for track in tracks:
            if track.track_id == config.start_track_id:
                return track
        raise GenerationError(f"Requested start track is not available: {config.start_track_id}")
    target = {"build": 0.35, "maintain": 0.60, "release": 0.85, "peak": 0.85}[config.energy_trajectory]
    return min(tracks, key=lambda track: (abs(track.energy.global_level - target), track.track_id))


def _assemble_set(
    renders: tuple[RenderResult, ...], output: Path, sample_rate: int, join_seconds: float
) -> dict[str, float | int]:
    parts = []
    for result in renders:
        samples, rate = sf.read(result.output_path, dtype="float32")
        if rate != sample_rate:
            raise GenerationError("Rendered segments have inconsistent sample rates")
        parts.append(samples)
    join_samples = int(round(join_seconds * sample_rate))
    assembled = parts[0]
    join_jumps: list[float] = []
    for part in parts[1:]:
        overlap = min(join_samples, assembled.size, part.size)
        if overlap == 0:
            join_jumps.append(float(abs(assembled[-1] - part[0])))
            assembled = np.concatenate((assembled, part))
            continue
        phase = np.linspace(0.0, np.pi / 2.0, overlap, endpoint=True, dtype=np.float32)
        mixed = assembled[-overlap:] * np.cos(phase) + part[:overlap] * np.sin(phase)
        before_edge = float(abs(assembled[-overlap - 1] - mixed[0])) if assembled.size > overlap else 0.0
        after_edge = float(abs(mixed[-1] - part[overlap])) if part.size > overlap else 0.0
        assembled = np.concatenate((assembled[:-overlap], mixed, part[overlap:]))
        join_jumps.append(max(before_edge, after_edge))
    if not np.isfinite(assembled).all():
        raise GenerationError("Rendered set contains invalid samples")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, assembled, sample_rate, format="WAV", subtype="FLOAT")
    return {
        "count": len(join_jumps),
        "duration_seconds": round(join_samples / sample_rate, 6),
        "max_boundary_jump": round(max(join_jumps, default=0.0), 6),
        "mean_boundary_jump": round(float(np.mean(join_jumps)) if join_jumps else 0.0, 6),
    }


def _validate_render(result: RenderResult) -> None:
    if result.clipping_samples or result.peak > 1.0 or result.duration <= 0.0:
        raise GenerationError(f"Rendered transition failed quality checks: {result.output_path}")


def _set_quality(
    path: Path, renders: list[RenderResult], joins: dict[str, float | int]
) -> dict[str, float | int | None | dict[str, float | int]]:
    samples, sample_rate = sf.read(path, dtype="float32")
    segment_rms = np.asarray([item.rms for item in renders], dtype=float)
    return {
        "sample_rate": int(sample_rate),
        "duration": round(samples.size / sample_rate, 6),
        "peak": round(float(np.max(np.abs(samples))), 6),
        "rms": round(float(np.sqrt(np.mean(samples**2))), 6),
        "clipping_samples": int(np.count_nonzero(np.abs(samples) > 1.0)),
        "finite": int(np.isfinite(samples).all()),
        "join_quality": joins,
        "transition_beat_alignment_max_seconds": round(max((abs(item.beat_alignment_seconds) for item in renders), default=0.0), 6),
        "transition_rms_range_db": round(float(20.0 * np.log10(max(segment_rms) / max(min(segment_rms), 1e-12))), 6),
        "artifact_rate": None,
    }


def _validate_set_quality(quality: dict[str, Any]) -> None:
    if quality["finite"] != 1 or quality["clipping_samples"] != 0 or quality["peak"] > 1.0:
        raise GenerationError("Final set failed the finite-sample or clipping quality gate")


def _write_human_report(path: Path, plan: SetPlan, tracks: dict[str, TrackAnalysis], metrics: dict[str, Any], fallbacks: list[str]) -> None:
    lines = ["AI DJ SET", "", "Tracks:"]
    lines.extend(f"{index:02d}. {Path(tracks[track_id].source_path).name}" for index, track_id in enumerate(plan.track_ids, start=1))
    lines.extend(["", "Transitions:"])
    for step in plan.steps:
        reasons = step.transition.strengths or tuple(component.reason for component in step.transition.components if component.score >= 0.75)
        lines.append(f"{step.transition.source_track_id} → {step.transition.destination_track_id}: {step.transition_score:.3f} ({step.transition.strategy})")
        lines.extend(f"  - {reason}" for reason in reasons[:3])
    lines.extend(["", f"Energy trajectory: {metrics['energy_trajectory']}", f"Fallbacks: {', '.join(fallbacks)}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_config(config: GenerationConfig) -> None:
    if config.target_track_count < 2:
        raise ValueError("target_track_count must be at least two")
    if config.candidate_pool_size < 1 or config.beam_width < 1 or config.sample_rate < 1 or config.set_join_seconds < 0.0:
        raise ValueError("candidate_pool_size, beam_width, and sample_rate must be positive; set_join_seconds cannot be negative")
