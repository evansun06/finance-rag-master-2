"""
Prepare deterministic per-video transcripts from the sentence panel.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import LOGGER, MATCHES_SENTENCE_PANEL_FILE, OUTPUT_DIR


DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "prompt-6" / "google_sentence_panel_batch1_prepared_transcripts.csv"
SORT_SENTINEL = 10**18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic transcripts from the sentence panel.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=MATCHES_SENTENCE_PANEL_FILE,
        help="Sentence-panel CSV to read.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Prepared transcript CSV to write.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, csv_path: Path) -> None:
    required = {"Text", "Sentence ID", "Onset", "VideoID"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns {missing}: {csv_path}")


def main() -> None:
    args = parse_args()
    LOGGER.info("Loading sentence panel from %s", args.input_csv)

    frame = pd.read_csv(
        args.input_csv,
        usecols=["Text", "Sentence ID", "Onset", "VideoID"],
    )
    require_columns(frame, args.input_csv)

    frame["_row_index"] = range(len(frame))
    frame["VideoID"] = frame["VideoID"].fillna("").astype(str).str.strip()
    frame = frame[frame["VideoID"] != ""].copy()
    frame["Text"] = frame["Text"].fillna("").astype(str).str.strip()
    frame["_sentence_sort"] = pd.to_numeric(frame["Sentence ID"], errors="coerce").fillna(SORT_SENTINEL)
    frame["_onset_sort"] = pd.to_numeric(frame["Onset"], errors="coerce").fillna(SORT_SENTINEL)

    LOGGER.info(
        "Preparing transcripts for %s videos from %s sentence rows.",
        f"{frame['VideoID'].nunique():,}",
        f"{len(frame):,}",
    )

    frame = frame.sort_values(
        by=["VideoID", "_sentence_sort", "_onset_sort", "_row_index"],
        kind="mergesort",
    )

    transcripts = (
        frame.groupby("VideoID", sort=True)["Text"]
        .agg(lambda series: " ".join(part for part in series if part))
        .reset_index(name="transcript_text")
        .rename(columns={"VideoID": "video_ID"})
    )
    transcripts = transcripts[transcripts["transcript_text"] != ""].copy()
    transcripts["text_length"] = transcripts["transcript_text"].str.len()
    transcripts = transcripts[["video_ID", "text_length", "transcript_text"]]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output_csv.parent,
        prefix=f".{args.output_csv.stem}.",
        suffix=args.output_csv.suffix,
        delete=False,
    ) as handle:
        transcripts.to_csv(handle.name, index=False)
        temp_path = Path(handle.name)

    temp_path.replace(args.output_csv)
    LOGGER.info(
        "Wrote %s prepared transcripts to %s",
        f"{len(transcripts):,}",
        args.output_csv,
    )


if __name__ == "__main__":
    main()
