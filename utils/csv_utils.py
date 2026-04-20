from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_rows_by_key(csv_path: Path, key_field: str) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}

    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get(key_field, "")
            if key:
                rows[key] = row
    return rows


def load_existing_results(csv_path: Path) -> dict[str, dict[str, str]]:
    return load_rows_by_key(csv_path, "video_ID")


def to_yes_no(value: Any) -> str:
    return "yes" if value else "no"


def _clean_json_payload(raw_response: Any) -> str:
    if raw_response is None:
        return ""
    if isinstance(raw_response, dict):
        return json.dumps(raw_response)
    if not isinstance(raw_response, str):
        return str(raw_response)

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    return cleaned


def parse_llm_response(raw_response: Any) -> dict[str, Any]:
    cleaned = _clean_json_payload(raw_response)
    if not cleaned:
        return {}

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def write_rows(
    csv_path: Path,
    fieldnames: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    sort_key: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=csv_path.parent,
        prefix=f".{csv_path.stem}.",
        suffix=csv_path.suffix,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: str(item.get(sort_key, ""))):
            writer.writerow(row)
        temp_path = Path(handle.name)

    temp_path.replace(csv_path)
