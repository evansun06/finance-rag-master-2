"""
Application of finance-academia backed RAG on the 49-video Odean set.
"""

from __future__ import annotations

import argparse
import csv
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


EXPECTED_VIDEO_COUNT = 49
CONTENT_INPUT_FILE = ODEAN_INPUT_DIR / "content_odean.csv"
WORD_PANEL_INPUT_FILE = ODEAN_INPUT_DIR / "word_panel_odean.csv"


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
    required_files = [CONTENT_INPUT_FILE, WORD_PANEL_INPUT_FILE]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file does not exist: {file_path}")


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def load_video_ids(csv_path: Path) -> list[str]:
    video_ids: list[str] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"Expected an 'id' column in {csv_path}")

        for row_number, row in enumerate(reader, start=2):
            video_id = (row.get("id") or "").strip()
            if not video_id:
                raise ValueError(f"Missing id in {csv_path} at row {row_number}")
            if video_id in seen:
                raise ValueError(f"Duplicate id '{video_id}' found in {csv_path}")
            seen.add(video_id)
            video_ids.append(video_id)

    if len(video_ids) != EXPECTED_VIDEO_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VIDEO_COUNT} unique ids in {csv_path}, found {len(video_ids)}"
        )

    return video_ids


def load_word_panel(csv_path: Path) -> dict[str, list[dict[str, str | int]]]:
    rows_by_video: dict[str, list[dict[str, str | int]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"Text", "Onset", "VideoID"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Word panel is missing one of the required columns: {sorted(required_columns)}")

        for row_index, row in enumerate(reader):
            video_id = (row.get("VideoID") or "").strip()
            if not video_id:
                continue

            rows_by_video.setdefault(video_id, []).append(
                {
                    "Text": row.get("Text", ""),
                    "Onset": row.get("Onset", ""),
                    "_row_index": row_index,
                }
            )

    return rows_by_video


def build_video_transcript(video_id: str, word_rows: list[dict[str, str | int]]) -> str:
    sorted_rows = sorted(
        word_rows,
        key=lambda row: (
            parse_float(str(row.get("Onset", ""))),
            int(row.get("_row_index", 0)),
        ),
    )

    parts = [str(row.get("Text", "")).strip() for row in sorted_rows]
    transcript_text = " ".join(part for part in parts if part)
    if not transcript_text:
        raise ValueError(f"No transcript text found for Odean video {video_id}")
    return transcript_text


def build_transcript_lookup(
    video_ids: list[str], rows_by_video: dict[str, list[dict[str, str | int]]]
) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for video_id in video_ids:
        word_rows = rows_by_video.get(video_id)
        if not word_rows:
            raise ValueError(f"Odean video {video_id} was not found in {WORD_PANEL_INPUT_FILE}")
        transcripts[video_id] = build_video_transcript(video_id, word_rows)
    return transcripts


def main() -> None:
    args = parse_args()
    require_configuration()

    video_ids = load_video_ids(CONTENT_INPUT_FILE)
    valid_video_ids = set(video_ids)
    rows_by_video = load_word_panel(WORD_PANEL_INPUT_FILE)
    transcript_lookup = build_transcript_lookup(video_ids, rows_by_video)

    retriever, chain = prepare_analysis_resources(args.rebuild_index)

    existing_results = load_existing_results(ODEAN_OUTPUT_FILE)
    results = {video_id: row for video_id, row in existing_results.items() if video_id in valid_video_ids}
    removed_count = len(existing_results) - len(results)
    if removed_count:
        LOGGER.info("Removing %s stale rows from %s before analysis.", removed_count, ODEAN_OUTPUT_FILE)

    total = len(video_ids)
    LOGGER.info("Found %s Odean videos in %s", total, CONTENT_INPUT_FILE)

    for index, video_id in enumerate(video_ids, start=1):
        if video_id in results and not args.overwrite:
            LOGGER.info("[%s/%s] Skipping %s (already processed).", index, total, video_id)
            continue

        transcript_text = transcript_lookup[video_id]
        LOGGER.info("[%s/%s] Analyzing %s (%s chars) ...", index, total, video_id, f"{len(transcript_text):,}")

        results[video_id] = analyze_transcript(video_id, transcript_text, retriever, chain)
        write_rows(ODEAN_OUTPUT_FILE, CSV_COLUMNS, results, sort_key="video_ID")

    LOGGER.info("Done. Results written to %s", ODEAN_OUTPUT_FILE)


if __name__ == "__main__":
    main()
