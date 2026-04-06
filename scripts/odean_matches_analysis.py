"""
Application of finance-acadmia backed RAG on 49 Best and 49 Worst matches
for the ODEAN video set.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    LOGGER,
    MATCHES_BEST_INPUT_FILE,
    MATCHES_BEST_OUTPUT_FILE,
    MATCHES_SENTENCE_PANEL_FILE,
    MATCHES_WORST_INPUT_FILE,
    MATCHES_WORST_OUTPUT_FILE,
)
from utils.csv_utils import load_existing_results, write_rows
from utils.odean_pipeline import (
    CSV_COLUMNS,
    analyze_transcript,
    prepare_analysis_resources,
    require_base_configuration,
)


EXPECTED_MATCH_COUNT = 49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze matched transcript sets against the finance RAG corpus.")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the local FAISS index before analysis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run analysis for transcript sets already present in the output CSV.",
    )
    return parser.parse_args()


def require_configuration() -> None:
    require_base_configuration()
    required_files = [
        MATCHES_BEST_INPUT_FILE,
        MATCHES_WORST_INPUT_FILE,
        MATCHES_SENTENCE_PANEL_FILE,
    ]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file does not exist: {file_path}")


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return sys.maxsize


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def load_match_video_ids(csv_path: Path) -> list[str]:
    video_ids: list[str] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "b1_id" not in (reader.fieldnames or []):
            raise ValueError(f"Expected a 'b1_id' column in {csv_path}")

        for row_number, row in enumerate(reader, start=2):
            video_id = (row.get("b1_id") or "").strip()
            if not video_id:
                raise ValueError(f"Missing b1_id in {csv_path} at row {row_number}")
            if video_id in seen:
                raise ValueError(f"Duplicate b1_id '{video_id}' found in {csv_path}")
            seen.add(video_id)
            video_ids.append(video_id)

    if len(video_ids) != EXPECTED_MATCH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MATCH_COUNT} unique b1_id values in {csv_path}, found {len(video_ids)}"
        )

    return video_ids


def load_sentence_panel(csv_path: Path) -> dict[str, list[dict[str, str | int]]]:
    rows_by_video: dict[str, list[dict[str, str | int]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"Text", "Sentence ID", "Onset", "VideoID"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Sentence panel is missing one of the required columns: {sorted(required_columns)}")

        for row_index, row in enumerate(reader):
            video_id = (row.get("VideoID") or "").strip()
            if not video_id:
                continue

            rows_by_video.setdefault(video_id, []).append(
                {
                    "Text": row.get("Text", ""),
                    "Sentence ID": row.get("Sentence ID", ""),
                    "Onset": row.get("Onset", ""),
                    "_row_index": row_index,
                }
            )

    return rows_by_video


def build_video_transcript(video_id: str, sentence_rows: list[dict[str, str | int]]) -> str:
    sorted_rows = sorted(
        sentence_rows,
        key=lambda row: (
            parse_int(str(row.get("Sentence ID", ""))),
            parse_float(str(row.get("Onset", ""))),
            int(row.get("_row_index", 0)),
        ),
    )

    parts = [str(row.get("Text", "")).strip() for row in sorted_rows]
    transcript_text = " ".join(part for part in parts if part)
    if not transcript_text:
        raise ValueError(f"No transcript text found for matched video {video_id}")
    return transcript_text


def build_transcript_lookup(
    video_ids: list[str], rows_by_video: dict[str, list[dict[str, str | int]]]
) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for video_id in video_ids:
        sentence_rows = rows_by_video.get(video_id)
        if not sentence_rows:
            raise ValueError(f"Matched video {video_id} was not found in {MATCHES_SENTENCE_PANEL_FILE}")
        transcripts[video_id] = build_video_transcript(video_id, sentence_rows)
    return transcripts


def main() -> None:
    args = parse_args()
    require_configuration()

    rows_by_video = load_sentence_panel(MATCHES_SENTENCE_PANEL_FILE)
    dataset_specs = [
        {
            "label": "best matched set",
            "match_file": MATCHES_BEST_INPUT_FILE,
            "output_file": MATCHES_BEST_OUTPUT_FILE,
        },
        {
            "label": "worst matched set",
            "match_file": MATCHES_WORST_INPUT_FILE,
            "output_file": MATCHES_WORST_OUTPUT_FILE,
        },
    ]

    retriever, chain = prepare_analysis_resources(args.rebuild_index)

    total = len(dataset_specs)
    for index, dataset in enumerate(dataset_specs, start=1):
        output_file = dataset["output_file"]
        video_ids = load_match_video_ids(dataset["match_file"])
        valid_video_ids = set(video_ids)
        existing_results = load_existing_results(output_file)
        results = {
            video_id: row for video_id, row in existing_results.items() if video_id in valid_video_ids
        }
        removed_count = len(existing_results) - len(results)
        if removed_count:
            LOGGER.info(
                "[%s/%s] Removing %s stale rows from %s before analysis.",
                index,
                total,
                removed_count,
                output_file,
            )
        transcript_lookup = build_transcript_lookup(video_ids, rows_by_video)
        LOGGER.info("[%s/%s] Found %s matched videos for %s.", index, total, len(video_ids), dataset["label"])

        dataset_total = len(video_ids)
        for dataset_index, video_id in enumerate(video_ids, start=1):
            if video_id in results and not args.overwrite:
                LOGGER.info(
                    "[%s/%s][%s/%s] Skipping %s (already processed).",
                    index,
                    total,
                    dataset_index,
                    dataset_total,
                    video_id,
                )
                continue

            transcript_text = transcript_lookup[video_id]
            LOGGER.info(
                "[%s/%s][%s/%s] Analyzing %s (%s chars) ...",
                index,
                total,
                dataset_index,
                dataset_total,
                video_id,
                f"{len(transcript_text):,}",
            )

            results[video_id] = analyze_transcript(video_id, transcript_text, retriever, chain)
            write_rows(output_file, CSV_COLUMNS, results, sort_key="video_ID")

        LOGGER.info("[%s/%s] Results written to %s", index, total, output_file)


if __name__ == "__main__":
    main()
