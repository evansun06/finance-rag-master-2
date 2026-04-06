"""
Application of finance-acadmia backed RAG on video transcripts.
Original Script for Odean Analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import LOGGER, ODEAN_INPUT_DIR, ODEAN_OUTPUT_FILE
from utils.csv_utils import load_existing_results, write_rows
from utils.odean_pipeline import (
    CSV_COLUMNS,
    analyze_transcript,
    prepare_analysis_resources,
    require_base_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Odean transcripts against the finance RAG corpus.")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the local FAISS index before analysis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run analysis for transcripts already present in the output CSV.",
    )
    return parser.parse_args()


def require_configuration() -> None:
    require_base_configuration()
    if not ODEAN_INPUT_DIR.exists():
        raise FileNotFoundError(f"Transcript input directory does not exist: {ODEAN_INPUT_DIR}")


def main() -> None:
    args = parse_args()
    require_configuration()

    ODEAN_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    retriever, chain = prepare_analysis_resources(args.rebuild_index)

    results = load_existing_results(ODEAN_OUTPUT_FILE)
    transcript_paths = sorted(ODEAN_INPUT_DIR.glob("*.txt"))
    if not transcript_paths:
        LOGGER.info("No transcript files found in %s", ODEAN_INPUT_DIR)
        return

    total = len(transcript_paths)
    LOGGER.info("Found %s transcript files in %s", total, ODEAN_INPUT_DIR)

    for index, transcript_path in enumerate(transcript_paths, start=1):
        video_id = transcript_path.stem
        if video_id in results and not args.overwrite:
            LOGGER.info("[%s/%s] Skipping %s (already processed).", index, total, video_id)
            continue

        transcript_text = transcript_path.read_text(encoding="utf-8")
        LOGGER.info("[%s/%s] Analyzing %s (%s chars) ...", index, total, video_id, f"{len(transcript_text):,}")

        results[video_id] = analyze_transcript(video_id, transcript_text, retriever, chain)
        write_rows(ODEAN_OUTPUT_FILE, CSV_COLUMNS, results, sort_key="video_ID")

    LOGGER.info("Done. Results written to %s", ODEAN_OUTPUT_FILE)


if __name__ == "__main__":
    main()
