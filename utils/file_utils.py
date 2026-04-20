from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_json_file(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)

    temp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def advisory_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "created_at": utc_now_iso(),
    }

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            existing = read_json_file(lock_path, default={}) or {}
            existing_pid = int(existing.get("pid") or 0)
            if existing_pid and _pid_is_running(existing_pid):
                raise RuntimeError(
                    f"Lock file already exists and process {existing_pid} is still running: {lock_path}"
                )
            lock_path.unlink(missing_ok=True)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)
