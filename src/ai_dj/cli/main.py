"""Command-line interface for local library analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ai_dj.ingestion.scanner import AudioDirectoryNotFoundError
from ai_dj.pipeline.analyze import analyze_library
from ai_dj.pipeline.cache import AnalysisCache
from ai_dj.representation.json_io import write_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai_dj", description="Analyze a local music library.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Scan, analyze, and cache supported audio files.")
    analyze.add_argument("directory", type=Path, help="Music directory to scan recursively.")
    analyze.add_argument("--cache-dir", type=Path, help="Directory for reusable cache entries.")
    analyze.add_argument("--output-dir", type=Path, help="Directory for plain TrackAnalysis JSON results.")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)-8s %(message)s", force=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.command != "analyze":  # pragma: no cover - argparse enforces the only current command.
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
