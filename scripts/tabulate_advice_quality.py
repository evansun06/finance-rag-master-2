"""
Tabulate advice-quality score counts for Odean, best-match, and worst-match CSVs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = REPO_ROOT / "data" / "out" / "advice_quality_distribution.csv"
VALID_SCORES = (1, 2, 3, 4, 5)
OUTPUT_COLUMNS = ("score", "odean", "best", "worst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count advice_quality scores from Odean, best-match, and worst-match result CSVs."
    )
    parser.add_argument("odean_csv", type=Path, help="CSV file for Odean results.")
    parser.add_argument("best_csv", type=Path, help="CSV file for best-match results.")
    parser.add_argument("worst_csv", type=Path, help="CSV file for worst-match results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT_FILE}",
    )
    return parser.parse_args()


def load_score_counts(csv_path: Path) -> dict[int, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")

    counts: Counter[int] = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "advice_quality" not in (reader.fieldnames or []):
            raise ValueError(f"Expected an 'advice_quality' column in {csv_path}")

        for row_number, row in enumerate(reader, start=2):
            raw_value = (row.get("advice_quality") or "").strip()
            if not raw_value:
                raise ValueError(f"Missing advice_quality in {csv_path} at row {row_number}")

            try:
                score = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid advice_quality '{raw_value}' in {csv_path} at row {row_number}; expected an integer 1-5"
                ) from exc

            if score not in VALID_SCORES:
                raise ValueError(
                    f"Invalid advice_quality '{raw_value}' in {csv_path} at row {row_number}; expected a score from 1 to 5"
                )

            counts[score] += 1

    return {score: counts.get(score, 0) for score in VALID_SCORES}


def write_distribution_csv(
    output_path: Path,
    odean_counts: dict[int, int],
    best_counts: dict[int, int],
    worst_counts: dict[int, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for score in VALID_SCORES:
            writer.writerow(
                {
                    "score": score,
                    "odean": odean_counts[score],
                    "best": best_counts[score],
                    "worst": worst_counts[score],
                }
            )


def main() -> None:
    args = parse_args()

    odean_counts = load_score_counts(args.odean_csv)
    best_counts = load_score_counts(args.best_csv)
    worst_counts = load_score_counts(args.worst_csv)

    write_distribution_csv(args.output, odean_counts, best_counts, worst_counts)
    print(f"Wrote advice-quality distribution to {args.output}")


if __name__ == "__main__":
    main()
