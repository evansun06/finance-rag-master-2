"""
Prompt-6 batch analysis pipeline for speaker-segment transcripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import LOGGER, OPENAI_API_KEY, OUTPUT_DIR
from utils.csv_utils import write_rows
from utils.file_utils import advisory_lock, atomic_write_json, atomic_write_text, read_json_file, utc_now_iso
from utils.odean_pipeline import CSV_COLUMNS, DOCUMENT_SEPARATOR, RETRIEVAL_K, RETRIEVAL_QUERY_CHARS, require_base_configuration
from utils.prompt6_batch import (
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
    build_prompt6_chat_completion_body,
    build_prompt6_result_row,
    parse_prompt6_chat_completion_response,
    prepare_prompt6_retriever,
    retrieve_context,
)


DATASET_NAME = "speakerseg_batch1_20260121_prompt6_segments"
BATCH_SIZE = 1000
BULK_EMBEDDING_CHUNK_SIZE = 256
COMPLETION_WINDOW = "24h"
DEFAULT_POLL_SECONDS = 60
DEFAULT_PREPARED_CSV = REPO_ROOT / "data" / "in" / "speakerseg_batch1_20260121.csv"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "prompt-6" / "speakerseg_batch1_20260121_segment_analysis_results.csv"
DEFAULT_STATE_DIR = OUTPUT_DIR / "prompt-6" / "speakerseg_batch1_20260121_segment_batch_state"
MANIFEST_VERSION = 1
MERGE_PARSER_SIGNATURE = "prompt6_segment_batch_merge_v1"
MIN_ANALYSIS_WORDS = 10
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
MERGEABLE_BATCH_STATUSES = {"completed", "expired"}
RESUBMITTABLE_BATCH_STATUSES = {"failed"}
SEGMENT_CSV_COLUMNS = ["video_ID", "segment_no", *CSV_COLUMNS[1:]]
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_CODES = {
    "billing_hard_limit_reached",
    "insufficient_quota",
    "internal_error",
    "rate_limit_exceeded",
    "server_error",
    "service_unavailable",
    "timeout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-6 segment analysis through the OpenAI Batch API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Prepare and submit any missing segments as batch jobs.")
    add_submit_args(submit_parser)

    poll_parser = subparsers.add_parser("poll", help="Poll remote batch jobs and merge completed segment results.")
    add_state_args(poll_parser)
    poll_parser.add_argument("--watch", action="store_true", help="Keep polling until all tracked jobs are terminal.")
    poll_parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Seconds between poll loops when --watch is used.",
    )

    run_parser = subparsers.add_parser("run", help="Submit missing segments, then poll batch jobs.")
    add_submit_args(run_parser)
    run_parser.add_argument("--watch", action="store_true", help="Keep polling until all tracked jobs are terminal.")
    run_parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Seconds between poll loops when --watch is used.",
    )

    status_parser = subparsers.add_parser("status", help="Show local batch-job status without calling the API.")
    add_state_args(status_parser)
    return parser.parse_args()


def add_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Final prompt-6-compatible segment output CSV.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Directory used for JSON manifests, batch request files, and downloaded outputs.",
    )


def add_submit_args(parser: argparse.ArgumentParser) -> None:
    add_state_args(parser)
    parser.add_argument(
        "--prepared-csv",
        type=Path,
        default=DEFAULT_PREPARED_CSV,
        help="Speaker-segment transcript CSV.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Submit segments even if they already exist in the final output CSV.",
    )
    output_mode.add_argument(
        "--retry-errors",
        action="store_true",
        help="Submit only retryable failed Batch API line items from downloaded error JSONL files.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the local FAISS index before local retrieval for request preparation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only submit the first N prepared segments after filtering and sorting.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=BULK_EMBEDDING_CHUNK_SIZE,
        help="Inputs per synchronous embeddings HTTP request while preparing RAG context.",
    )


def require_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")


def manifest_path(state_dir: Path) -> Path:
    return state_dir / "manifest.json"


def batches_dir(state_dir: Path) -> Path:
    return state_dir / "batches"


def requests_dir(state_dir: Path) -> Path:
    return state_dir / "requests"


def downloads_dir(state_dir: Path) -> Path:
    return state_dir / "downloads"


def lock_path(state_dir: Path) -> Path:
    return state_dir / "runner.lock"


def batch_state_path(state_dir: Path, local_batch_id: str) -> Path:
    return batches_dir(state_dir) / f"{local_batch_id}.json"


def ensure_state_dirs(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    batches_dir(state_dir).mkdir(parents=True, exist_ok=True)
    requests_dir(state_dir).mkdir(parents=True, exist_ok=True)
    downloads_dir(state_dir).mkdir(parents=True, exist_ok=True)


def build_manifest(prepared_csv: Path, output_csv: Path, embedding_batch_size: int) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": DATASET_NAME,
        "prepared_csv": str(prepared_csv),
        "output_csv": str(output_csv),
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "batch_size": BATCH_SIZE,
        "min_analysis_words": MIN_ANALYSIS_WORDS,
        "embedding_batch_size": embedding_batch_size,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "local_batch_ids": [],
    }


def load_manifest(state_dir: Path, prepared_csv: Path, output_csv: Path, embedding_batch_size: int) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    existing = read_json_file(manifest_path(state_dir))
    if existing:
        existing["prepared_csv"] = str(prepared_csv)
        existing["output_csv"] = str(output_csv)
        existing["min_analysis_words"] = MIN_ANALYSIS_WORDS
        existing["embedding_batch_size"] = embedding_batch_size
        existing["updated_at"] = utc_now_iso()
        atomic_write_json(manifest_path(state_dir), existing)
        return existing
    manifest = build_manifest(prepared_csv, output_csv, embedding_batch_size)
    atomic_write_json(manifest_path(state_dir), manifest)
    return manifest


def save_manifest(state_dir: Path, manifest: dict[str, Any], batch_states: Iterable[dict[str, Any]]) -> None:
    manifest["updated_at"] = utc_now_iso()
    manifest["local_batch_ids"] = [batch_state["local_batch_id"] for batch_state in sorted(batch_states, key=batch_sort_key)]
    atomic_write_json(manifest_path(state_dir), manifest)


def batch_sort_key(batch_state: dict[str, Any]) -> tuple[int, str]:
    local_batch_id = str(batch_state.get("local_batch_id") or "")
    try:
        numeric = int(local_batch_id.split("-")[-1])
    except ValueError:
        numeric = sys.maxsize
    return numeric, local_batch_id


def load_batch_states(state_dir: Path) -> list[dict[str, Any]]:
    if not batches_dir(state_dir).exists():
        return []
    states = [
        read_json_file(path)
        for path in sorted(batches_dir(state_dir).glob("*.json"))
    ]
    return [state for state in states if state]


def save_batch_state(state_dir: Path, batch_state: dict[str, Any]) -> None:
    batch_state["updated_at"] = utc_now_iso()
    atomic_write_json(batch_state_path(state_dir, batch_state["local_batch_id"]), batch_state)


def increase_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def build_segment_key(video_id: str, segment_no: str) -> str:
    return f"{video_id}::segment::{segment_no}"


def segment_word_count(transcript_text: str) -> int:
    return len(transcript_text.split())


def meets_segment_analysis_threshold(transcript_text: str) -> bool:
    return segment_word_count(transcript_text) >= MIN_ANALYSIS_WORDS


def load_prepared_segments(prepared_csv: Path) -> list[dict[str, str]]:
    increase_csv_field_size_limit()
    rows: list[dict[str, str]] = []
    with prepared_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"VideoID", "segment_no", "transcript"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Segment CSV is missing columns {missing}: {prepared_csv}")
        for row in reader:
            video_id = (row.get("VideoID") or "").strip()
            segment_no = (row.get("segment_no") or "").strip()
            if video_id and segment_no:
                transcript_text = str(row.get("transcript") or "")
                rows.append(
                    {
                        "segment_key": build_segment_key(video_id, segment_no),
                        "video_ID": video_id,
                        "segment_no": segment_no,
                        "text_length": str(len(transcript_text)),
                        "transcript_text": transcript_text,
                    }
                )
    return rows


def load_prepared_lookup(prepared_csv: Path) -> dict[str, dict[str, str]]:
    return {row["segment_key"]: row for row in load_prepared_segments(prepared_csv)}


def load_existing_segment_results(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}

    increase_csv_field_size_limit()
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = str(row.get("video_ID") or "").strip()
            segment_no = str(row.get("segment_no") or "").strip()
            if video_id and segment_no:
                rows[build_segment_key(video_id, segment_no)] = row
    return rows


def chunk_rows(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def next_local_batch_id(batch_states: list[dict[str, Any]]) -> str:
    if not batch_states:
        return "batch-0001"
    highest = max(batch_sort_key(batch_state)[0] for batch_state in batch_states)
    return f"batch-{highest + 1:04d}"


def sync_batch_state_from_remote(batch_state: dict[str, Any], remote_batch: Any) -> dict[str, Any]:
    remote = remote_batch.to_dict()
    batch_state["openai_batch_id"] = remote.get("id")
    batch_state["status"] = remote.get("status", batch_state.get("status", "unknown"))
    batch_state["input_file_id"] = remote.get("input_file_id") or batch_state.get("input_file_id")
    batch_state["output_file_id"] = remote.get("output_file_id")
    batch_state["error_file_id"] = remote.get("error_file_id")
    batch_state["request_counts"] = remote.get("request_counts") or batch_state.get("request_counts") or {}
    batch_state["remote_metadata"] = remote.get("metadata") or {}
    for field in (
        "created_at",
        "in_progress_at",
        "completed_at",
        "failed_at",
        "expired_at",
        "expires_at",
    ):
        if remote.get(field) is not None:
            batch_state[field] = remote.get(field)
    batch_state["last_polled_at"] = utc_now_iso()
    return batch_state


def build_draft_batch_state(
    state_dir: Path,
    local_batch_id: str,
    prepared_csv: Path,
    rows: list[dict[str, str]],
    document_names_by_segment: dict[str, list[str]],
    retry_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_state = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": DATASET_NAME,
        "local_batch_id": local_batch_id,
        "submission_token": str(uuid.uuid4()),
        "status": "draft",
        "merge_status": "pending",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "prepared_csv": str(prepared_csv),
        "input_jsonl_path": str(requests_dir(state_dir) / f"{local_batch_id}.jsonl"),
        "downloaded_output_path": str(downloads_dir(state_dir) / f"{local_batch_id}.output.jsonl"),
        "downloaded_error_path": str(downloads_dir(state_dir) / f"{local_batch_id}.error.jsonl"),
        "segment_keys": [row["segment_key"] for row in rows],
        "video_ids": [row["video_ID"] for row in rows],
        "segment_count": len(rows),
        "video_count": len(rows),
        "document_names_by_segment": document_names_by_segment,
        "request_counts": {
            "total": len(rows),
            "completed": 0,
            "failed": 0,
        },
    }
    if retry_metadata:
        batch_state.update(retry_metadata)
    return batch_state


def build_batch_metadata(batch_state: dict[str, Any]) -> dict[str, str]:
    metadata = {
        "dataset_name": DATASET_NAME,
        "local_batch_id": batch_state["local_batch_id"],
        "submission_token": batch_state["submission_token"],
    }
    for field in ("retry_mode", "retry_source_count", "retry_candidate_count", "retry_error_codes"):
        value = batch_state.get(field)
        if value is not None:
            if isinstance(value, (list, tuple, set)):
                metadata[field] = ",".join(str(item) for item in value)
            else:
                metadata[field] = str(value)
    return metadata


def load_reserved_segment_keys(batch_states: list[dict[str, Any]]) -> set[str]:
    reserved: set[str] = set()
    for batch_state in batch_states:
        status = str(batch_state.get("status") or "")
        if batch_state.get("merge_status") == "merged":
            continue
        if status in RESUBMITTABLE_BATCH_STATUSES:
            continue
        reserved.update(batch_state.get("segment_keys") or [])
    return reserved


def write_request_jsonl(batch_state: dict[str, Any], request_lines: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(line, ensure_ascii=True) for line in request_lines) + "\n"
    atomic_write_text(Path(batch_state["input_jsonl_path"]), text)


def recover_remote_batch(client: OpenAI, submission_token: str) -> Any | None:
    page = client.batches.list(limit=100)
    for batch in page.data:
        metadata = getattr(batch, "metadata", None) or {}
        if metadata.get("submission_token") == submission_token:
            return batch
    return None


def submit_or_resume_batch(client: OpenAI, state_dir: Path, batch_state: dict[str, Any]) -> dict[str, Any]:
    if batch_state.get("openai_batch_id"):
        remote_batch = client.batches.retrieve(batch_state["openai_batch_id"])
        batch_state = sync_batch_state_from_remote(batch_state, remote_batch)
        save_batch_state(state_dir, batch_state)
        return batch_state

    recovered = recover_remote_batch(client, batch_state["submission_token"])
    if recovered is not None:
        LOGGER.info(
            "Recovered remote batch %s for %s using submission token.",
            recovered.id,
            batch_state["local_batch_id"],
        )
        batch_state = sync_batch_state_from_remote(batch_state, recovered)
        save_batch_state(state_dir, batch_state)
        return batch_state

    if not batch_state.get("input_file_id"):
        LOGGER.info("Uploading input JSONL for %s", batch_state["local_batch_id"])
        with Path(batch_state["input_jsonl_path"]).open("rb") as handle:
            upload = client.files.create(file=handle, purpose="batch")
        batch_state["input_file_id"] = upload.id
        batch_state["status"] = "input_uploaded"
        save_batch_state(state_dir, batch_state)

    LOGGER.info(
        "Creating OpenAI batch job for %s (%s segments)",
        batch_state["local_batch_id"],
        batch_state["segment_count"],
    )
    remote_batch = client.batches.create(
        input_file_id=batch_state["input_file_id"],
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
        metadata=build_batch_metadata(batch_state),
    )
    batch_state = sync_batch_state_from_remote(batch_state, remote_batch)
    save_batch_state(state_dir, batch_state)
    LOGGER.info(
        "Created remote batch %s for %s with status %s",
        batch_state["openai_batch_id"],
        batch_state["local_batch_id"],
        batch_state["status"],
    )
    return batch_state


def ensure_downloaded_file(client: OpenAI, file_id: str | None, target_path: Path) -> Path | None:
    if not file_id:
        return None
    if target_path.exists():
        return target_path
    response = client.files.content(file_id)
    atomic_write_text(target_path, response.text)
    return target_path


def load_jsonl_by_custom_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            custom_id = str(item.get("custom_id") or "").strip()
            if custom_id:
                payload[custom_id] = item
    return payload


def iter_jsonl_items(path: Path | None) -> Iterable[dict[str, Any]]:
    if path is None or not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSONL row in %s:%s: %s", path, line_no, exc)
                continue
            if isinstance(item, dict):
                yield item


def batch_line_status_code(item: dict[str, Any]) -> int | None:
    response = item.get("response") or {}
    status_code = response.get("status_code")
    if status_code is None:
        return None
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def extract_batch_line_error(item: dict[str, Any]) -> dict[str, Any]:
    response = item.get("response") or {}
    body = response.get("body") or {}
    error = item.get("error") or response.get("error") or body.get("error") or {}
    if isinstance(error, dict):
        return error
    if error:
        return {"message": str(error)}
    return {}


def batch_line_error_code(item: dict[str, Any]) -> str:
    error = extract_batch_line_error(item)
    return str(error.get("code") or error.get("type") or "").strip().lower()


def describe_batch_line_error(item: dict[str, Any]) -> str:
    status_code = batch_line_status_code(item)
    error = extract_batch_line_error(item)
    code = error.get("code") or error.get("type")
    message = error.get("message")
    pieces = []
    if status_code is not None:
        pieces.append(f"status={status_code}")
    if code:
        pieces.append(f"code={code}")
    if message:
        pieces.append(f"message={message}")
    return ", ".join(pieces) if pieces else "unknown error"


def is_retryable_batch_line_error(item: dict[str, Any]) -> bool:
    status_code = batch_line_status_code(item)
    error_code = batch_line_error_code(item)
    return status_code in RETRYABLE_HTTP_STATUS_CODES or error_code in RETRYABLE_ERROR_CODES


def local_batch_sort_key(local_batch_id: str) -> tuple[int, str]:
    return batch_sort_key({"local_batch_id": local_batch_id})


def local_batch_id_from_download_path(path: Path) -> str | None:
    name = path.name
    for suffix in (".error.jsonl", ".output.jsonl", "-error.jsonl", "-output.jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def discover_download_local_batch_ids(state_dir: Path) -> set[str]:
    local_batch_ids: set[str] = set()
    for pattern in ("batch-*.error.jsonl", "batch-*-error.jsonl", "batch-*.output.jsonl", "batch-*-output.jsonl"):
        for path in downloads_dir(state_dir).glob(pattern):
            local_batch_id = local_batch_id_from_download_path(path)
            if local_batch_id:
                local_batch_ids.add(local_batch_id)
    return local_batch_ids


def downloaded_jsonl_paths(
    state_dir: Path,
    local_batch_id: str,
    kind: str,
    batch_state: dict[str, Any] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if batch_state:
        configured_path = batch_state.get(f"downloaded_{kind}_path")
        if configured_path:
            paths.append(Path(configured_path))
    paths.append(downloads_dir(state_dir) / f"{local_batch_id}.{kind}.jsonl")
    paths.append(downloads_dir(state_dir) / f"{local_batch_id}-{kind}.jsonl")

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def load_unmerged_segment_keys(batch_states: list[dict[str, Any]]) -> set[str]:
    reserved: set[str] = set()
    for batch_state in batch_states:
        if batch_state.get("merge_status") != "merged":
            reserved.update(batch_state.get("segment_keys") or [])
    return reserved


def batch_state_needs_submission_resume(batch_state: dict[str, Any]) -> bool:
    if batch_state.get("merge_status") == "merged":
        return False
    status = str(batch_state.get("status") or "")
    return bool(batch_state.get("openai_batch_id")) or status in {"draft", "input_uploaded"}


def has_resumable_batch_states(batch_states: list[dict[str, Any]]) -> bool:
    return any(batch_state_needs_submission_resume(batch_state) for batch_state in batch_states)


def resume_existing_batch_submissions(
    client: OpenAI,
    state_dir: Path,
    batch_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resumed_states: list[dict[str, Any]] = []
    for batch_state in sorted(batch_states, key=batch_sort_key):
        if batch_state_needs_submission_resume(batch_state):
            batch_state = submit_or_resume_batch(client, state_dir, batch_state)
        resumed_states.append(batch_state)
    return resumed_states


def collect_retry_error_segment_keys(
    state_dir: Path,
    batch_states: list[dict[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    batch_by_id = {
        str(batch_state.get("local_batch_id")): batch_state
        for batch_state in batch_states
        if batch_state.get("local_batch_id")
    }
    local_batch_ids = set(batch_by_id) | discover_download_local_batch_ids(state_dir)
    latest_attempts: dict[str, dict[str, Any]] = {}
    retry_source_paths: set[str] = set()
    retryable_error_codes: set[str] = set()
    retryable_error_rows = 0
    successful_rows = 0

    for local_batch_id in sorted(local_batch_ids, key=local_batch_sort_key):
        batch_state = batch_by_id.get(local_batch_id)

        for path in downloaded_jsonl_paths(state_dir, local_batch_id, "error", batch_state):
            retry_source_paths.add(str(path))
            for item in iter_jsonl_items(path):
                segment_key = str(item.get("custom_id") or "").strip()
                if not segment_key:
                    continue
                error_code = batch_line_error_code(item)
                if is_retryable_batch_line_error(item):
                    retryable_error_rows += 1
                    if error_code:
                        retryable_error_codes.add(error_code)
                    latest_attempts[segment_key] = {
                        "status": "retryable_error",
                        "local_batch_id": local_batch_id,
                        "source_path": str(path),
                        "error_code": error_code,
                    }
                else:
                    latest_attempts[segment_key] = {
                        "status": "non_retryable_error",
                        "local_batch_id": local_batch_id,
                        "source_path": str(path),
                        "error_code": error_code,
                    }

        for path in downloaded_jsonl_paths(state_dir, local_batch_id, "output", batch_state):
            for item in iter_jsonl_items(path):
                segment_key = str(item.get("custom_id") or "").strip()
                if not segment_key:
                    continue
                status_code = batch_line_status_code(item)
                if status_code == 200:
                    successful_rows += 1
                    latest_attempts[segment_key] = {
                        "status": "success",
                        "local_batch_id": local_batch_id,
                        "source_path": str(path),
                    }
                elif is_retryable_batch_line_error(item):
                    error_code = batch_line_error_code(item)
                    retryable_error_rows += 1
                    if error_code:
                        retryable_error_codes.add(error_code)
                    latest_attempts[segment_key] = {
                        "status": "retryable_error",
                        "local_batch_id": local_batch_id,
                        "source_path": str(path),
                        "error_code": error_code,
                    }
                else:
                    latest_attempts[segment_key] = {
                        "status": "non_retryable_error",
                        "local_batch_id": local_batch_id,
                        "source_path": str(path),
                        "error_code": batch_line_error_code(item),
                    }

    unmerged_segment_keys = load_unmerged_segment_keys(batch_states)
    retry_segment_keys = {
        segment_key
        for segment_key, attempt in latest_attempts.items()
        if attempt.get("status") == "retryable_error" and segment_key not in unmerged_segment_keys
    }
    summary = {
        "retry_source_file_count": len(retry_source_paths),
        "retryable_error_rows": retryable_error_rows,
        "successful_output_rows": successful_rows,
        "latest_retryable_segment_count": sum(
            1 for attempt in latest_attempts.values() if attempt.get("status") == "retryable_error"
        ),
        "unmerged_segment_count": len(unmerged_segment_keys),
        "retryable_error_codes": sorted(retryable_error_codes),
    }
    return retry_segment_keys, summary


def select_retry_error_rows(
    prepared_rows: list[dict[str, str]],
    state_dir: Path,
    batch_states: list[dict[str, Any]],
    limit: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    retry_segment_keys, summary = collect_retry_error_segment_keys(state_dir, batch_states)
    prepared_keys = {row["segment_key"] for row in prepared_rows}
    unknown_retry_keys = retry_segment_keys - prepared_keys

    retry_rows = [row for row in prepared_rows if row["segment_key"] in retry_segment_keys]
    if limit is not None:
        retry_rows = retry_rows[:limit]

    short_rows = [
        row
        for row in retry_rows
        if not meets_segment_analysis_threshold(row["transcript_text"])
    ]
    pending_analysis_rows = [
        row
        for row in retry_rows
        if meets_segment_analysis_threshold(row["transcript_text"])
    ]
    summary["unknown_retry_key_count"] = len(unknown_retry_keys)
    summary["selected_retry_row_count"] = len(retry_rows)
    summary["selected_retry_analysis_count"] = len(pending_analysis_rows)
    summary["selected_retry_short_count"] = len(short_rows)
    return pending_analysis_rows, short_rows, summary


def build_segment_result_row(
    video_id: str,
    segment_no: str,
    transcript_text: str,
    document_names: list[str],
    parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = build_prompt6_result_row(
        video_id=video_id,
        transcript_text=transcript_text,
        document_names=document_names,
        parsed=parsed,
    )
    row["segment_no"] = segment_no
    return row


def build_short_segment_null_row(row: dict[str, str]) -> dict[str, Any]:
    output_row = {column: "" for column in SEGMENT_CSV_COLUMNS}
    output_row["video_ID"] = row["video_ID"]
    output_row["segment_no"] = row["segment_no"]
    output_row["text_length"] = row["text_length"]
    return output_row


def write_short_segment_null_rows(output_csv: Path, rows: list[dict[str, str]]) -> int:
    if not rows:
        return 0

    results = load_existing_segment_results(output_csv)
    for row in rows:
        results[row["segment_key"]] = build_short_segment_null_row(row)
    write_rows(output_csv, SEGMENT_CSV_COLUMNS, results, sort_key="video_ID")
    return len(rows)


def format_retrieved_documents(documents: list[Any]) -> tuple[str, list[str]]:
    context_pieces = []
    document_names = []
    for document in documents:
        page_content = document.page_content.strip()
        if page_content:
            context_pieces.append(page_content)

        metadata = getattr(document, "metadata", {})
        document_name = metadata.get("title") or metadata.get("source") or "unknown-document"
        document_names.append(str(document_name))

    return DOCUMENT_SEPARATOR.join(context_pieces), document_names


def embedding_queries_for_rows(rows: list[dict[str, str]]) -> list[str]:
    return [row["transcript_text"][:RETRIEVAL_QUERY_CHARS] for row in rows]


def retrieve_contexts_by_embedding_vectors(
    retriever: Any,
    rows: list[dict[str, str]],
    embedding_batch_size: int,
    local_batch_id: str,
) -> dict[str, tuple[str, list[str]]]:
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is None:
        raise ValueError("Retriever does not expose a vectorstore for bulk retrieval.")

    embedding_function = getattr(vectorstore, "embedding_function", None)
    if embedding_function is None or not hasattr(embedding_function, "embed_documents"):
        raise ValueError("Vectorstore does not expose an embed_documents-compatible embedding function.")

    search_kwargs = dict(getattr(retriever, "search_kwargs", {}) or {})
    k = int(search_kwargs.pop("k", RETRIEVAL_K))
    contexts: dict[str, tuple[str, list[str]]] = {}

    for chunk_start in range(0, len(rows), embedding_batch_size):
        chunk_rows = rows[chunk_start : chunk_start + embedding_batch_size]
        chunk_contexts = retrieve_contexts_for_chunk(
            retriever=retriever,
            vectorstore=vectorstore,
            embedding_function=embedding_function,
            rows=chunk_rows,
            k=k,
            search_kwargs=search_kwargs,
        )
        contexts.update(chunk_contexts)
        LOGGER.info(
            "Prepared RAG context for %s/%s segments in %s using bulk embeddings",
            min(chunk_start + len(chunk_rows), len(rows)),
            len(rows),
            local_batch_id,
        )

    return contexts


def retrieve_contexts_for_chunk(
    retriever: Any,
    vectorstore: Any,
    embedding_function: Any,
    rows: list[dict[str, str]],
    k: int,
    search_kwargs: dict[str, Any],
) -> dict[str, tuple[str, list[str]]]:
    if not rows:
        return {}

    try:
        embeddings = embedding_function.embed_documents(embedding_queries_for_rows(rows), chunk_size=len(rows))
        contexts: dict[str, tuple[str, list[str]]] = {}
        for row, embedding in zip(rows, embeddings, strict=True):
            documents = vectorstore.max_marginal_relevance_search_by_vector(
                embedding,
                k=k,
                **search_kwargs,
            )
            contexts[row["segment_key"]] = format_retrieved_documents(documents)
        return contexts
    except Exception as exc:
        if len(rows) > 1:
            midpoint = len(rows) // 2
            LOGGER.warning(
                "Bulk retrieval failed for %s rows; splitting chunk and retrying: %s",
                len(rows),
                exc,
            )
            left = retrieve_contexts_for_chunk(
                retriever=retriever,
                vectorstore=vectorstore,
                embedding_function=embedding_function,
                rows=rows[:midpoint],
                k=k,
                search_kwargs=search_kwargs,
            )
            right = retrieve_contexts_for_chunk(
                retriever=retriever,
                vectorstore=vectorstore,
                embedding_function=embedding_function,
                rows=rows[midpoint:],
                k=k,
                search_kwargs=search_kwargs,
            )
            return {**left, **right}

        row = rows[0]
        try:
            return {row["segment_key"]: retrieve_context(retriever, row["transcript_text"])}
        except Exception as fallback_exc:
            LOGGER.warning("Retrieval failed for %s during submit: %s", row["segment_key"], fallback_exc)
            return {row["segment_key"]: ("", [])}


def merge_completed_batch(
    client: OpenAI,
    batch_state: dict[str, Any],
    prepared_lookup: dict[str, dict[str, str]],
    output_csv: Path,
) -> dict[str, Any]:
    output_path = ensure_downloaded_file(
        client,
        batch_state.get("output_file_id"),
        Path(batch_state["downloaded_output_path"]),
    )
    error_path = ensure_downloaded_file(
        client,
        batch_state.get("error_file_id"),
        Path(batch_state["downloaded_error_path"]),
    )

    output_map = load_jsonl_by_custom_id(output_path)
    error_map = load_jsonl_by_custom_id(error_path)
    results = load_existing_segment_results(output_csv)
    fallback_count = 0

    for segment_key in batch_state.get("segment_keys") or []:
        prepared = prepared_lookup.get(segment_key, {})
        video_id = prepared.get("video_ID", "")
        segment_no = prepared.get("segment_no", "")
        transcript_text = prepared.get("transcript_text", "")
        document_names = (batch_state.get("document_names_by_segment") or {}).get(segment_key, [])
        parsed: dict[str, Any] = {}

        output_item = output_map.get(segment_key)
        if output_item:
            response = output_item.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}
            if status_code == 200:
                try:
                    parsed = parse_prompt6_chat_completion_response(body)
                except Exception as exc:
                    LOGGER.warning("Response parsing failed for %s: %s", segment_key, exc)
                    parsed = {}
            else:
                LOGGER.warning(
                    "Non-200 response for %s in %s: %s",
                    segment_key,
                    batch_state["local_batch_id"],
                    status_code,
                )

        if not parsed and segment_key in error_map:
            LOGGER.warning(
                "Batch error for %s in %s: %s",
                segment_key,
                batch_state["local_batch_id"],
                describe_batch_line_error(error_map[segment_key]),
            )

        if not parsed:
            fallback_count += 1

        results[segment_key] = build_segment_result_row(
            video_id=video_id,
            segment_no=segment_no,
            transcript_text=transcript_text,
            document_names=document_names,
            parsed=parsed or None,
        )

    write_rows(output_csv, SEGMENT_CSV_COLUMNS, results, sort_key="video_ID")
    batch_state["merge_status"] = "merged"
    batch_state["merged_at"] = utc_now_iso()
    batch_state["merged_row_count"] = len(batch_state.get("segment_keys") or [])
    batch_state["fallback_row_count"] = fallback_count
    batch_state["merge_parser_signature"] = MERGE_PARSER_SIGNATURE
    LOGGER.info(
        "Merged %s rows from %s into %s (%s fallback rows).",
        batch_state["merged_row_count"],
        batch_state["local_batch_id"],
        output_csv,
        fallback_count,
    )
    return batch_state


def poll_once(client: OpenAI, state_dir: Path, output_csv: Path, prepared_lookup: dict[str, dict[str, str]]) -> bool:
    batch_states = load_batch_states(state_dir)
    if not batch_states:
        LOGGER.info("No batch-state JSON files found in %s", state_dir)
        return True

    all_terminal = True
    merged_this_pass = 0

    for batch_state in sorted(batch_states, key=batch_sort_key):
        local_batch_id = batch_state["local_batch_id"]
        if batch_state.get("openai_batch_id"):
            remote_batch = client.batches.retrieve(batch_state["openai_batch_id"])
            batch_state = sync_batch_state_from_remote(batch_state, remote_batch)
            save_batch_state(state_dir, batch_state)
        elif batch_state.get("submission_token"):
            recovered = recover_remote_batch(client, batch_state["submission_token"])
            if recovered is not None:
                batch_state = sync_batch_state_from_remote(batch_state, recovered)
                save_batch_state(state_dir, batch_state)

        counts = batch_state.get("request_counts") or {}
        LOGGER.info(
            "Batch %s remote_id=%s status=%s completed=%s failed=%s total=%s merged=%s",
            local_batch_id,
            batch_state.get("openai_batch_id", "-"),
            batch_state.get("status", "unknown"),
            counts.get("completed", 0),
            counts.get("failed", 0),
            counts.get("total", batch_state.get("segment_count", 0)),
            batch_state.get("merge_status", "pending"),
        )

        status = str(batch_state.get("status") or "")
        if status not in TERMINAL_BATCH_STATUSES:
            all_terminal = False
            continue

        if status in MERGEABLE_BATCH_STATUSES and (
            batch_state.get("merge_status") != "merged"
            or batch_state.get("merge_parser_signature") != MERGE_PARSER_SIGNATURE
        ):
            batch_state = merge_completed_batch(client, batch_state, prepared_lookup, output_csv)
            save_batch_state(state_dir, batch_state)
            merged_this_pass += 1
        elif status == "failed":
            LOGGER.warning(
                "Batch %s failed validation or creation remotely. Its segments were not merged and may need resubmission.",
                local_batch_id,
            )

    final_row_count = len(load_existing_segment_results(output_csv))
    LOGGER.info(
        "Poll pass complete. Merged %s batch files this pass. Final CSV rows now at %s.",
        merged_this_pass,
        final_row_count,
    )
    return all_terminal


def print_local_status(state_dir: Path, output_csv: Path) -> None:
    batch_states = load_batch_states(state_dir)
    existing_results = load_existing_segment_results(output_csv)
    LOGGER.info("Local final CSV rows: %s", len(existing_results))
    LOGGER.info("Tracked local batch manifests: %s", len(batch_states))

    aggregate_completed = 0
    aggregate_failed = 0
    aggregate_total = 0
    aggregate_merged_rows = 0
    for batch_state in batch_states:
        counts = batch_state.get("request_counts") or {}
        aggregate_completed += int(counts.get("completed") or 0)
        aggregate_failed += int(counts.get("failed") or 0)
        aggregate_total += int(counts.get("total") or batch_state.get("segment_count") or 0)
        if batch_state.get("merge_status") == "merged":
            aggregate_merged_rows += int(batch_state.get("merged_row_count") or 0)

    LOGGER.info(
        "Last-known remote totals: completed=%s failed=%s total=%s merged_rows=%s",
        aggregate_completed,
        aggregate_failed,
        aggregate_total,
        aggregate_merged_rows,
    )

    for batch_state in sorted(batch_states, key=batch_sort_key):
        counts = batch_state.get("request_counts") or {}
        LOGGER.info(
            "Batch %s remote_id=%s status=%s merge=%s completed=%s failed=%s total=%s last_polled_at=%s",
            batch_state.get("local_batch_id"),
            batch_state.get("openai_batch_id", "-"),
            batch_state.get("status", "unknown"),
            batch_state.get("merge_status", "pending"),
            counts.get("completed", 0),
            counts.get("failed", 0),
            counts.get("total", batch_state.get("segment_count", 0)),
            batch_state.get("last_polled_at", "-"),
        )


def submit_analysis_batches(
    args: argparse.Namespace,
    client: OpenAI,
    manifest: dict[str, Any],
    batch_states: list[dict[str, Any]],
    pending_analysis_rows: list[dict[str, str]],
    retry_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    retriever = prepare_prompt6_retriever(args.rebuild_index)

    for batch_rows in chunk_rows(pending_analysis_rows, BATCH_SIZE):
        local_batch_id = next_local_batch_id(batch_states)
        request_lines: list[dict[str, Any]] = []
        document_names_by_segment: dict[str, list[str]] = {}
        contexts_by_segment = retrieve_contexts_by_embedding_vectors(
            retriever=retriever,
            rows=batch_rows,
            embedding_batch_size=args.embedding_batch_size,
            local_batch_id=local_batch_id,
        )

        for row_index, row in enumerate(batch_rows, start=1):
            segment_key = row["segment_key"]
            transcript_text = row["transcript_text"]
            context, document_names = contexts_by_segment.get(segment_key, ("", []))

            document_names_by_segment[segment_key] = document_names
            request_lines.append(
                {
                    "custom_id": segment_key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_prompt6_chat_completion_body(context, transcript_text),
                }
            )

            if row_index == len(batch_rows) or row_index % 10 == 0:
                LOGGER.info(
                    "Prepared %s/%s segments for %s",
                    row_index,
                    len(batch_rows),
                    local_batch_id,
                )

        batch_state = build_draft_batch_state(
            state_dir=args.state_dir,
            local_batch_id=local_batch_id,
            prepared_csv=args.prepared_csv,
            rows=batch_rows,
            document_names_by_segment=document_names_by_segment,
            retry_metadata=retry_metadata,
        )
        write_request_jsonl(batch_state, request_lines)
        save_batch_state(args.state_dir, batch_state)
        batch_states.append(batch_state)
        save_manifest(args.state_dir, manifest, batch_states)
        batch_state = submit_or_resume_batch(client, args.state_dir, batch_state)
        batch_states[-1] = batch_state
        save_manifest(args.state_dir, manifest, batch_states)

    return batch_states


def submit_batches(args: argparse.Namespace) -> None:
    if args.embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size must be at least 1.")

    ensure_state_dirs(args.state_dir)
    manifest = load_manifest(args.state_dir, args.prepared_csv, args.output_csv, args.embedding_batch_size)
    prepared_rows = load_prepared_segments(args.prepared_csv)

    batch_states = load_batch_states(args.state_dir)
    client: OpenAI | None = None

    if has_resumable_batch_states(batch_states):
        require_base_configuration()
        require_openai_key()
        client = OpenAI(api_key=OPENAI_API_KEY)
        batch_states = resume_existing_batch_submissions(client, args.state_dir, batch_states)
        save_manifest(args.state_dir, manifest, batch_states)

    retry_metadata: dict[str, Any] | None = None
    if args.retry_errors:
        pending_analysis_rows, short_rows, retry_summary = select_retry_error_rows(
            prepared_rows=prepared_rows,
            state_dir=args.state_dir,
            batch_states=batch_states,
            limit=args.limit,
        )
        retry_metadata = {
            "retry_mode": "batch_errors",
            "retry_source_count": retry_summary["retry_source_file_count"],
            "retry_candidate_count": retry_summary["selected_retry_analysis_count"],
            "retry_error_codes": retry_summary["retryable_error_codes"],
        }
        LOGGER.info(
            "Retry scan found %s latest retryable segments from %s retryable error rows in %s error files "
            "(%s successful output rows observed, %s active/unmerged segments excluded, %s unknown keys skipped).",
            retry_summary["latest_retryable_segment_count"],
            retry_summary["retryable_error_rows"],
            retry_summary["retry_source_file_count"],
            retry_summary["successful_output_rows"],
            retry_summary["unmerged_segment_count"],
            retry_summary["unknown_retry_key_count"],
        )
    else:
        existing_results = {} if args.overwrite else load_existing_segment_results(args.output_csv)
        reserved_segment_keys = load_reserved_segment_keys(batch_states)
        pending_rows = [
            row
            for row in prepared_rows
            if row["segment_key"] not in existing_results and row["segment_key"] not in reserved_segment_keys
        ]
        if args.limit is not None:
            pending_rows = pending_rows[: args.limit]

        short_rows = [
            row
            for row in pending_rows
            if not meets_segment_analysis_threshold(row["transcript_text"])
        ]
        pending_analysis_rows = [
            row
            for row in pending_rows
            if meets_segment_analysis_threshold(row["transcript_text"])
        ]

    short_row_count = write_short_segment_null_rows(args.output_csv, short_rows)
    if short_row_count:
        LOGGER.info(
            "Wrote %s too-short segment null rows to %s (threshold: >=%s words).",
            short_row_count,
            args.output_csv,
            MIN_ANALYSIS_WORDS,
        )

    if not pending_analysis_rows:
        if args.retry_errors:
            LOGGER.info(
                "No retryable analyzable error rows need submission. selected_retry_rows=%s too_short_null_rows=%s",
                retry_metadata["retry_candidate_count"] if retry_metadata else 0,
                short_row_count,
            )
        else:
            LOGGER.info(
                "No analyzable segments need submission. Existing rows=%s reserved=%s too_short_null_rows=%s",
                len(existing_results),
                len(reserved_segment_keys),
                short_row_count,
            )
        save_manifest(args.state_dir, manifest, batch_states)
        return

    require_base_configuration()
    require_openai_key()
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)

    if args.retry_errors:
        LOGGER.info(
            "Preparing %s retry-error segments for batch submission (%s too-short rows filled).",
            len(pending_analysis_rows),
            short_row_count,
        )
    else:
        LOGGER.info(
            "Preparing %s segments for batch submission (%s existing rows skipped, %s reserved segments skipped, %s too-short rows filled).",
            len(pending_analysis_rows),
            len(existing_results),
            len(reserved_segment_keys),
            short_row_count,
        )

    batch_states = submit_analysis_batches(
        args=args,
        client=client,
        manifest=manifest,
        batch_states=batch_states,
        pending_analysis_rows=pending_analysis_rows,
        retry_metadata=retry_metadata,
    )
    save_manifest(args.state_dir, manifest, batch_states)


def poll_batches(args: argparse.Namespace) -> None:
    require_openai_key()
    ensure_state_dirs(args.state_dir)
    manifest = read_json_file(manifest_path(args.state_dir))
    if not manifest:
        LOGGER.info("No manifest found in %s", args.state_dir)
        return

    prepared_csv = Path(manifest["prepared_csv"])
    prepared_lookup = load_prepared_lookup(prepared_csv)
    client = OpenAI(api_key=OPENAI_API_KEY)

    while True:
        all_terminal = poll_once(client, args.state_dir, args.output_csv, prepared_lookup)
        batch_states = load_batch_states(args.state_dir)
        save_manifest(args.state_dir, manifest, batch_states)
        if not getattr(args, "watch", False) or all_terminal:
            break
        LOGGER.info("Sleeping %s seconds before the next poll pass.", args.poll_seconds)
        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()

    if args.command == "status":
        print_local_status(args.state_dir, args.output_csv)
        return

    with advisory_lock(lock_path(args.state_dir)):
        if args.command == "submit":
            submit_batches(args)
            return
        if args.command == "poll":
            poll_batches(args)
            return
        if args.command == "run":
            submit_batches(args)
            poll_batches(args)
            return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
