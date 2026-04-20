"""
Prompt-6 batch analysis pipeline for the prepared sentence-panel transcripts.
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
from utils.csv_utils import load_existing_results, write_rows
from utils.file_utils import advisory_lock, atomic_write_json, atomic_write_text, read_json_file, utc_now_iso
from utils.odean_pipeline import CSV_COLUMNS, require_base_configuration
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


DATASET_NAME = "google_sentence_panel_batch1_prompt6"
BATCH_SIZE = 100
COMPLETION_WINDOW = "24h"
DEFAULT_POLL_SECONDS = 60
DEFAULT_PREPARED_CSV = OUTPUT_DIR / "prompt-6" / "google_sentence_panel_batch1_prepared_transcripts.csv"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "prompt-6" / "google_sentence_panel_batch1_analysis_results.csv"
DEFAULT_STATE_DIR = OUTPUT_DIR / "prompt-6" / "google_sentence_panel_batch1_batch_state"
MANIFEST_VERSION = 1
MERGE_PARSER_SIGNATURE = "prompt6_batch_merge_v2"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
MERGEABLE_BATCH_STATUSES = {"completed", "expired"}
RESUBMITTABLE_BATCH_STATUSES = {"failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-6 analysis through the OpenAI Batch API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Prepare and submit any missing videos as batch jobs.")
    add_submit_args(submit_parser)

    poll_parser = subparsers.add_parser("poll", help="Poll remote batch jobs and merge completed results.")
    add_state_args(poll_parser)
    poll_parser.add_argument("--watch", action="store_true", help="Keep polling until all tracked jobs are terminal.")
    poll_parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Seconds between poll loops when --watch is used.",
    )

    run_parser = subparsers.add_parser("run", help="Submit missing videos, then poll batch jobs.")
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
        help="Final prompt-6-compatible output CSV.",
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
        help="Prepared transcript CSV from prepare_sentence_panel_transcripts.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Submit videos even if they already exist in the final output CSV.",
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
        help="Only submit the first N prepared transcripts after filtering and sorting.",
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


def build_manifest(prepared_csv: Path, output_csv: Path) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": DATASET_NAME,
        "prepared_csv": str(prepared_csv),
        "output_csv": str(output_csv),
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "batch_size": BATCH_SIZE,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "local_batch_ids": [],
    }


def load_manifest(state_dir: Path, prepared_csv: Path, output_csv: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    existing = read_json_file(manifest_path(state_dir))
    if existing:
        existing["prepared_csv"] = str(prepared_csv)
        existing["output_csv"] = str(output_csv)
        existing["updated_at"] = utc_now_iso()
        atomic_write_json(manifest_path(state_dir), existing)
        return existing
    manifest = build_manifest(prepared_csv, output_csv)
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


def load_prepared_transcripts(prepared_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with prepared_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_ID", "text_length", "transcript_text"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Prepared transcript CSV is missing columns {missing}: {prepared_csv}")
        for row in reader:
            video_id = (row.get("video_ID") or "").strip()
            if video_id:
                rows.append(
                    {
                        "video_ID": video_id,
                        "text_length": str(row.get("text_length") or ""),
                        "transcript_text": str(row.get("transcript_text") or ""),
                    }
                )
    return rows


def load_prepared_lookup(prepared_csv: Path) -> dict[str, dict[str, str]]:
    return {row["video_ID"]: row for row in load_prepared_transcripts(prepared_csv)}


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
    document_names_by_video: dict[str, list[str]],
) -> dict[str, Any]:
    return {
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
        "video_ids": [row["video_ID"] for row in rows],
        "video_count": len(rows),
        "document_names_by_video": document_names_by_video,
        "request_counts": {
            "total": len(rows),
            "completed": 0,
            "failed": 0,
        },
    }


def build_batch_metadata(batch_state: dict[str, Any]) -> dict[str, str]:
    return {
        "dataset_name": DATASET_NAME,
        "local_batch_id": batch_state["local_batch_id"],
        "submission_token": batch_state["submission_token"],
    }


def load_reserved_video_ids(batch_states: list[dict[str, Any]]) -> set[str]:
    reserved: set[str] = set()
    for batch_state in batch_states:
        status = str(batch_state.get("status") or "")
        if batch_state.get("merge_status") == "merged":
            continue
        if status in RESUBMITTABLE_BATCH_STATUSES:
            continue
        reserved.update(batch_state.get("video_ids") or [])
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
        "Creating OpenAI batch job for %s (%s videos)",
        batch_state["local_batch_id"],
        batch_state["video_count"],
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
    results = load_existing_results(output_csv)
    fallback_count = 0

    for video_id in batch_state.get("video_ids") or []:
        prepared = prepared_lookup.get(video_id, {})
        transcript_text = prepared.get("transcript_text", "")
        document_names = (batch_state.get("document_names_by_video") or {}).get(video_id, [])
        parsed: dict[str, Any] = {}

        output_item = output_map.get(video_id)
        if output_item:
            response = output_item.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}
            if status_code == 200:
                try:
                    parsed = parse_prompt6_chat_completion_response(body)
                except Exception as exc:
                    LOGGER.warning("Response parsing failed for %s: %s", video_id, exc)
                    parsed = {}
            else:
                LOGGER.warning(
                    "Non-200 response for %s in %s: %s",
                    video_id,
                    batch_state["local_batch_id"],
                    status_code,
                )

        if not parsed and video_id in error_map:
            LOGGER.warning(
                "Batch error for %s in %s: %s",
                video_id,
                batch_state["local_batch_id"],
                error_map[video_id].get("error"),
            )

        if not parsed:
            fallback_count += 1

        results[video_id] = build_prompt6_result_row(
            video_id=video_id,
            transcript_text=transcript_text,
            document_names=document_names,
            parsed=parsed or None,
        )

    write_rows(output_csv, CSV_COLUMNS, results, sort_key="video_ID")
    batch_state["merge_status"] = "merged"
    batch_state["merged_at"] = utc_now_iso()
    batch_state["merged_row_count"] = len(batch_state.get("video_ids") or [])
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
            counts.get("total", batch_state.get("video_count", 0)),
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
                "Batch %s failed validation or creation remotely. Its videos were not merged and may need resubmission.",
                local_batch_id,
            )

    final_row_count = len(load_existing_results(output_csv))
    LOGGER.info(
        "Poll pass complete. Merged %s batch files this pass. Final CSV rows now at %s.",
        merged_this_pass,
        final_row_count,
    )
    return all_terminal


def print_local_status(state_dir: Path, output_csv: Path) -> None:
    batch_states = load_batch_states(state_dir)
    existing_results = load_existing_results(output_csv)
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
        aggregate_total += int(counts.get("total") or batch_state.get("video_count") or 0)
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
            counts.get("total", batch_state.get("video_count", 0)),
            batch_state.get("last_polled_at", "-"),
        )


def submit_batches(args: argparse.Namespace) -> None:
    require_base_configuration()
    ensure_state_dirs(args.state_dir)
    require_openai_key()
    manifest = load_manifest(args.state_dir, args.prepared_csv, args.output_csv)
    prepared_rows = load_prepared_transcripts(args.prepared_csv)

    batch_states = load_batch_states(args.state_dir)
    existing_results = {} if args.overwrite else load_existing_results(args.output_csv)
    reserved_video_ids = load_reserved_video_ids(batch_states)
    pending_rows = [
        row
        for row in prepared_rows
        if row["video_ID"] not in existing_results and row["video_ID"] not in reserved_video_ids
    ]
    if args.limit is not None:
        pending_rows = pending_rows[: args.limit]
    if not pending_rows:
        LOGGER.info("No videos need submission. Existing rows=%s reserved=%s", len(existing_results), len(reserved_video_ids))
        save_manifest(args.state_dir, manifest, batch_states)
        return

    LOGGER.info(
        "Preparing %s videos for batch submission (%s existing rows skipped, %s reserved videos skipped).",
        len(pending_rows),
        len(existing_results),
        len(reserved_video_ids),
    )

    retriever = prepare_prompt6_retriever(args.rebuild_index)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Resume any local drafts before creating new ones.
    resumed_states: list[dict[str, Any]] = []
    for batch_state in sorted(batch_states, key=batch_sort_key):
        if batch_state.get("merge_status") == "merged":
            resumed_states.append(batch_state)
            continue
        status = str(batch_state.get("status") or "")
        if batch_state.get("openai_batch_id") or status in {"draft", "input_uploaded"}:
            batch_state = submit_or_resume_batch(client, args.state_dir, batch_state)
        resumed_states.append(batch_state)
    batch_states = resumed_states

    for batch_rows in chunk_rows(pending_rows, BATCH_SIZE):
        local_batch_id = next_local_batch_id(batch_states)
        request_lines: list[dict[str, Any]] = []
        document_names_by_video: dict[str, list[str]] = {}

        for row_index, row in enumerate(batch_rows, start=1):
            video_id = row["video_ID"]
            transcript_text = row["transcript_text"]
            try:
                context, document_names = retrieve_context(retriever, transcript_text)
            except Exception as exc:
                LOGGER.warning("Retrieval failed for %s during submit: %s", video_id, exc)
                context, document_names = "", []

            document_names_by_video[video_id] = document_names
            request_lines.append(
                {
                    "custom_id": video_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_prompt6_chat_completion_body(context, transcript_text),
                }
            )

            if row_index == len(batch_rows) or row_index % 10 == 0:
                LOGGER.info(
                    "Prepared %s/%s videos for %s",
                    row_index,
                    len(batch_rows),
                    local_batch_id,
                )

        batch_state = build_draft_batch_state(
            state_dir=args.state_dir,
            local_batch_id=local_batch_id,
            prepared_csv=args.prepared_csv,
            rows=batch_rows,
            document_names_by_video=document_names_by_video,
        )
        write_request_jsonl(batch_state, request_lines)
        save_batch_state(args.state_dir, batch_state)
        batch_states.append(batch_state)
        save_manifest(args.state_dir, manifest, batch_states)
        batch_state = submit_or_resume_batch(client, args.state_dir, batch_state)
        batch_states[-1] = batch_state
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
