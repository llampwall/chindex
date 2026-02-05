"""
Repo-level status file for dashboard integration.

Writes .chinvex-status.json to each repo being ingested, allowing
external tools (like allmind dashboard) to track ingestion status.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def write_repo_status(
    repo_path: Path,
    status: str,
    context: str,
    ingest_type: str | None = None,
    files_processed: int | None = None,
    error: str | None = None,
    started_at: str | None = None,
) -> None:
    """
    Write .chinvex-status.json to a repo directory.

    Args:
        repo_path: Path to the repository
        status: One of: "ingesting", "idle", "error"
        context: Chinvex context name
        ingest_type: "full" or "delta" (only when status="ingesting")
        files_processed: Number of files processed (only when status="idle")
        error: Error message (only when status="error")
        started_at: ISO timestamp when ingestion started (preserved from start to finish)
    """
    status_file = repo_path / ".chinvex-status.json"

    # Read existing status to preserve started_at if needed
    existing_started_at = started_at
    if not existing_started_at and status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                existing_started_at = existing.get("started_at")
        except Exception:
            pass

    data = {
        "status": status,
        "context": context,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if status == "ingesting":
        data["pid"] = os.getpid()
        data["started_at"] = existing_started_at or datetime.now(timezone.utc).isoformat()
        if ingest_type:
            data["type"] = ingest_type
    elif status == "idle":
        data["finished_at"] = datetime.now(timezone.utc).isoformat()
        if existing_started_at:
            data["started_at"] = existing_started_at
        if files_processed is not None:
            data["files_processed"] = files_processed
    elif status == "error":
        data["finished_at"] = datetime.now(timezone.utc).isoformat()
        if existing_started_at:
            data["started_at"] = existing_started_at
        if error:
            data["error"] = error

    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        # Don't fail ingest if status write fails
        import logging
        logging.getLogger(__name__).warning(f"Failed to write repo status: {e}")
