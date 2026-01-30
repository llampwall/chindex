"""Archive tier implementation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from .storage import Storage
from .util import now_iso

log = logging.getLogger(__name__)


@dataclass
class ArchiveStats:
    """Statistics from archive operation."""
    archived_count: int
    total_chunks: int
    active_chunks: int


def get_doc_age_timestamp(row: dict) -> datetime | None:
    """
    Get the timestamp used for archive age calculation.

    Uses updated_at for content age.
    """
    ts_str = row.get("updated_at")
    if not ts_str:
        return None

    # Parse ISO8601 timestamp
    try:
        # Remove 'Z' suffix if present
        ts_str = ts_str.rstrip("Z")
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def archive_old_documents(storage: Storage, age_threshold_days: int, dry_run: bool = False) -> int:
    """
    Archive documents older than threshold.

    Returns count of documents archived (or would be archived in dry-run).
    """
    from datetime import timezone

    threshold_date_naive = datetime.utcnow() - timedelta(days=age_threshold_days)
    threshold_date_aware = datetime.now(timezone.utc) - timedelta(days=age_threshold_days)

    # Find candidates
    cursor = storage.conn.execute(
        """
        SELECT doc_id, updated_at
        FROM documents
        WHERE archived = 0
        """
    )

    candidates = []
    for row in cursor.fetchall():
        row_dict = dict(row)
        doc_age = get_doc_age_timestamp(row_dict)
        if doc_age:
            # Use appropriate threshold based on whether doc_age is timezone-aware
            threshold = threshold_date_aware if doc_age.tzinfo is not None else threshold_date_naive
            if doc_age < threshold:
                candidates.append(row_dict["doc_id"])

    if dry_run:
        return len(candidates)

    # Execute archive
    if candidates:
        archived_at = now_iso()
        placeholders = ",".join("?" * len(candidates))
        storage._execute(
            f"""
            UPDATE documents
            SET archived = 1, archived_at = ?
            WHERE doc_id IN ({placeholders})
            """,
            (archived_at, *candidates)
        )
        storage.conn.commit()

    return len(candidates)


def list_archived_documents(storage: Storage, limit: int = 50) -> list[dict]:
    """
    List archived documents.

    Returns list of archived document metadata.
    """
    cursor = storage.conn.execute(
        """
        SELECT doc_id, source_type, title, archived_at
        FROM documents
        WHERE archived = 1
        ORDER BY archived_at DESC
        LIMIT ?
        """,
        (limit,)
    )

    return [dict(row) for row in cursor.fetchall()]


def restore_document(storage: Storage, doc_id: str) -> bool:
    """
    Restore archived document.

    Flips archived flag to 0. Does NOT re-ingest or re-embed.
    Returns True if document was found and restored.
    """
    cursor = storage.conn.execute(
        "SELECT archived FROM documents WHERE doc_id = ?",
        (doc_id,)
    )
    row = cursor.fetchone()

    if not row:
        return False

    if row["archived"] == 0:
        return False  # Already active

    storage._execute(
        "UPDATE documents SET archived = 0, archived_at = NULL WHERE doc_id = ?",
        (doc_id,)
    )
    storage.conn.commit()

    return True


def purge_archived_documents(storage: Storage, age_threshold_days: int, dry_run: bool = False) -> int:
    """
    Permanently delete archived documents older than threshold.

    Only deletes documents that are ALREADY archived.
    Returns count of documents purged.
    """
    from datetime import timezone

    threshold_date_naive = datetime.utcnow() - timedelta(days=age_threshold_days)
    threshold_date_aware = datetime.now(timezone.utc) - timedelta(days=age_threshold_days)

    # Find candidates (archived docs older than threshold)
    cursor = storage.conn.execute(
        """
        SELECT doc_id, archived_at
        FROM documents
        WHERE archived = 1 AND archived_at IS NOT NULL
        """
    )

    candidates = []
    for row in cursor.fetchall():
        archived_at_str = row["archived_at"].rstrip("Z")
        try:
            archived_at = datetime.fromisoformat(archived_at_str)

            # Use appropriate threshold based on whether archived_at is timezone-aware
            threshold = threshold_date_aware if archived_at.tzinfo is not None else threshold_date_naive

            if archived_at < threshold:
                candidates.append(row["doc_id"])
        except ValueError:
            continue

    if dry_run:
        return len(candidates)

    # Execute purge (delete from both documents and chunks)
    if candidates:
        placeholders = ",".join("?" * len(candidates))

        # Delete chunks first (foreign key constraint)
        storage._execute(
            f"DELETE FROM chunks WHERE doc_id IN ({placeholders})",
            candidates
        )

        # Delete documents
        storage._execute(
            f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
            candidates
        )

        storage.conn.commit()

    return len(candidates)


def archive_by_age(storage: Storage, age_threshold_days: int) -> ArchiveStats:
    """
    Archive chunks older than threshold.

    Args:
        storage: Storage instance
        age_threshold_days: Archive chunks older than this many days

    Returns:
        ArchiveStats with counts
    """
    threshold_dt = datetime.now(timezone.utc) - timedelta(days=age_threshold_days)
    threshold_str = threshold_dt.isoformat()

    # Ensure archived column exists
    _ensure_archived_column(storage)

    # Count total and active chunks before
    total_before = storage.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    active_before = storage.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE archived = 0"
    ).fetchone()[0]

    # Archive old chunks (only those not already archived)
    cursor = storage.conn.execute(
        """
        UPDATE chunks
        SET archived = 1
        WHERE updated_at < ?
          AND archived = 0
        """,
        (threshold_str,)
    )

    archived_count = cursor.rowcount
    storage.conn.commit()

    log.info(f"Archived {archived_count} chunks older than {age_threshold_days} days")

    return ArchiveStats(
        archived_count=archived_count,
        total_chunks=total_before,
        active_chunks=active_before - archived_count
    )


def archive_by_count(storage: Storage, max_chunks: int) -> ArchiveStats:
    """
    Archive oldest chunks to keep total under max_chunks.

    Args:
        storage: Storage instance
        max_chunks: Maximum active chunks to keep

    Returns:
        ArchiveStats with counts
    """
    _ensure_archived_column(storage)

    # Count active chunks
    active_count = storage.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE archived = 0"
    ).fetchone()[0]

    total_count = storage.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    if active_count <= max_chunks:
        log.info(f"Active chunks ({active_count}) under limit ({max_chunks})")
        return ArchiveStats(
            archived_count=0,
            total_chunks=total_count,
            active_chunks=active_count
        )

    # Archive oldest chunks
    to_archive = active_count - max_chunks

    cursor = storage.conn.execute(
        """
        UPDATE chunks
        SET archived = 1
        WHERE chunk_id IN (
            SELECT chunk_id FROM chunks
            WHERE archived = 0
            ORDER BY updated_at ASC
            LIMIT ?
        )
        """,
        (to_archive,)
    )

    archived_count = cursor.rowcount
    storage.conn.commit()

    log.info(f"Archived {archived_count} oldest chunks to stay under {max_chunks}")

    return ArchiveStats(
        archived_count=archived_count,
        total_chunks=total_count,
        active_chunks=max_chunks
    )


def _ensure_archived_column(storage: Storage) -> None:
    """Ensure archived column exists in chunks table."""
    # Check if column exists
    cursor = storage.conn.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in cursor.fetchall()]

    if "archived" not in columns:
        log.info("Adding archived column to chunks table")
        storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
        storage.conn.commit()
