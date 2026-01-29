from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_run_start(
    log_path: Path,
    run_id: str,
    sources: list[str]
) -> None:
    """Log the start of an ingest run."""
    entry = {
        "run_id": run_id,
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources": sources
    }

    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def log_run_end(
    log_path: Path,
    run_id: str,
    status: str,  # "succeeded" or "failed"
    docs_seen: int = 0,
    docs_changed: int = 0,
    chunks_new: int = 0,
    chunks_updated: int = 0,
    error: str | None = None
) -> None:
    """Log the end of an ingest run."""
    entry = {
        "run_id": run_id,
        "status": status,
        "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }

    if status == "succeeded":
        entry.update({
            "docs_seen": docs_seen,
            "docs_changed": docs_changed,
            "chunks_new": chunks_new,
            "chunks_updated": chunks_updated
        })
    elif status == "failed":
        entry["error"] = error

    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_ingest_runs(
    log_path: Path,
    completed_only: bool = False
) -> list[dict]:
    """
    Read ingest runs from JSONL log.
    If completed_only=True, return only end records (succeeded/failed), excluding start records.
    """
    if not log_path.exists():
        return []

    runs = []
    with log_path.open("r") as f:
        for line in f:
            runs.append(json.loads(line.strip()))

    if not completed_only:
        return runs

    # Filter: keep only end records (succeeded/failed status)
    return [r for r in runs if r["status"] in ("succeeded", "failed")]
