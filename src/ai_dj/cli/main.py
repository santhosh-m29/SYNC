"""Command-line interface for local library analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ai_dj.ingestion.scanner import AudioDirectoryNotFoundError
from ai_dj.pipeline.analyze import analyze_library
from ai_dj.pipeline.cache import AnalysisCache
from ai_dj.pipeline.generate import GenerationConfig, GenerationError, generate_dj_set
from ai_dj.representation.json_io import write_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai_dj", description="Analyze a local music library.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Scan, analyze, and cache supported audio files.")
    analyze.add_argument("directory", type=Path, help="Music directory to scan recursively.")
    analyze.add_argument("--cache-dir", type=Path, help="Directory for reusable cache entries.")
    analyze.add_argument("--output-dir", type=Path, help="Directory for plain TrackAnalysis JSON results.")
    generate = commands.add_parser("generate", help="Analyze, plan, render, and report a deterministic DJ-set preview.")
    generate.add_argument("--input", required=True, type=Path, help="Music directory to scan recursively.")
    generate.add_argument("--output", required=True, type=Path, help="Lossless WAV set output path.")
    generate.add_argument("--tracks", type=int, default=4, help="Target number of tracks (default: 4).")
    generate.add_argument("--trajectory", choices=("build", "maintain", "release", "peak"), default="maintain")
    generate.add_argument("--candidate-pool-size", type=int, default=24, help="Maximum cheap-prefiltered candidates.")
    generate.add_argument("--beam-width", type=int, default=8)
    generate.add_argument("--seed", type=int, default=7, help="Recorded reproducibility seed.")
    generate.add_argument("--cache-dir", type=Path, help="Directory for reusable analysis cache entries.")
    generate.add_argument("--analysis-output-dir", type=Path, help="Optional directory for TrackAnalysis JSON documents.")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)-8s %(message)s", force=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.command == "generate":
        try:
            result = generate_dj_set(
                args.input.expanduser(),
                args.output.expanduser(),
                GenerationConfig(
                    target_track_count=args.tracks,
                    energy_trajectory=args.trajectory,
                    candidate_pool_size=args.candidate_pool_size,
                    beam_width=args.beam_width,
                    seed=args.seed,
                ),
                cache_directory=args.cache_dir,
                analysis_output_directory=args.analysis_output_dir,
            )
        except (AudioDirectoryNotFoundError, GenerationError, ValueError) as error:
            logging.getLogger(__name__).error("Generation failed: %s", error)
            return 2
        logger = logging.getLogger(__name__)
        logger.info("Generated %s", result.output_path)
        logger.info("Report: %s", result.report_path)
        logger.info(
            "Tracks: %d; transitions: %d; average score: %.3f; technical failures: %d",
            result.metrics["tracks_selected"], result.metrics["transitions_generated"],
            result.metrics["average_transition_score"], result.metrics["technical_failures"],
        )
        return 0
    if args.command != "analyze":  # pragma: no cover - argparse enforces subcommands.
        return 2

    directory = args.directory.expanduser()
    cache_dir = args.cache_dir or directory / ".ai_dj_cache"
    output_dir = args.output_dir or directory / ".ai_dj_analysis"
    try:
        result = analyze_library(directory, AnalysisCache(cache_dir))
    except AudioDirectoryNotFoundError as error:
        logging.getLogger(__name__).error("%s", error)
        return 2

    for analysis in result.analyses:
        write_analysis(output_dir / f"{analysis.track_id}.json", analysis)
    logger = logging.getLogger(__name__)
    logger.info(
        "Analysis complete. Successful: %d; Failed: %d; Skipped: %d; Cached: %d",
        len(result.analyses),
        len(result.failures),
        result.skipped,
        result.cached,
    )
    return 1 if result.failures else 0
