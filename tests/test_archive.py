# tests/test_archive.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from chinvex.archive import archive_by_age, archive_by_count, ArchiveStats


def test_archive_by_age_marks_old_chunks(tmp_path: Path):
    """Should mark chunks older than threshold as archived"""
    from chinvex.storage import Storage
    from chinvex.util import now_iso

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Add archived column if not exists
    storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
    storage.conn.commit()

    # Insert chunks with different ages
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    recent_time = datetime.now(timezone.utc) - timedelta(days=10)

    # Use upsert_chunks with correct signature (13 columns):
    # chunk_id, doc_id, source_type, project, repo, chinvex_depth, status, tags_json, ordinal, text, updated_at, meta_json, chunk_key
    storage.upsert_chunks([
        ("chunk1", "doc1", "repo", "test", "test", "full", "active", "[]", 0, "old content", old_time.isoformat(), "{}", "key1"),
        ("chunk2", "doc1", "repo", "test", "test", "full", "active", "[]", 1, "recent content", recent_time.isoformat(), "{}", "key2"),
    ])

    # Archive chunks older than 90 days
    stats = archive_by_age(storage, age_threshold_days=90)

    assert stats.archived_count == 1

    # Verify chunk1 is archived
    result = storage.conn.execute(
        "SELECT archived FROM chunks WHERE chunk_id = ?", ("chunk1",)
    ).fetchone()
    assert result[0] == 1

    # Verify chunk2 is NOT archived
    result = storage.conn.execute(
        "SELECT archived FROM chunks WHERE chunk_id = ?", ("chunk2",)
    ).fetchone()
    assert result[0] == 0

    storage.close()


def test_archive_by_count_archives_oldest(tmp_path: Path):
    """Should archive oldest chunks when count exceeds limit"""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
    storage.conn.commit()

    # Insert 10 chunks with different ages
    chunks = []
    # 13 columns: chunk_id, doc_id, source_type, project, repo, chinvex_depth, status, tags_json, ordinal, text, updated_at, meta_json, chunk_key
    for i in range(10):
        age = datetime.now(timezone.utc) - timedelta(days=i)
        chunks.append((f"chunk{i}", "doc1", "repo", "test", "test", "full", "active", "[]", i, f"content{i}", age.isoformat(), "{}", f"key{i}"))
    storage.upsert_chunks(chunks)

    # Archive to keep only 5 chunks
    stats = archive_by_count(storage, max_chunks=5)

    assert stats.archived_count == 5

    # Verify oldest 5 are archived
    for i in range(5, 10):  # Oldest
        result = storage.conn.execute(
            "SELECT archived FROM chunks WHERE chunk_id = ?", (f"chunk{i}",)
        ).fetchone()
        assert result[0] == 1

    # Verify newest 5 are NOT archived
    for i in range(5):  # Newest
        result = storage.conn.execute(
            "SELECT archived FROM chunks WHERE chunk_id = ?", (f"chunk{i}",)
        ).fetchone()
        assert result[0] == 0

    storage.close()


def test_archive_skips_already_archived(tmp_path: Path):
    """Should not double-archive already archived chunks"""
    from chinvex.storage import Storage
    from chinvex.util import now_iso

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
    storage.conn.commit()

    # Insert chunk and mark as archived
    # 13 columns: chunk_id, doc_id, source_type, project, repo, chinvex_depth, status, tags_json, ordinal, text, updated_at, meta_json, chunk_key
    storage.upsert_chunks([("chunk1", "doc1", "repo", "test", "test", "full", "active", "[]", 0, "content", now_iso(), "{}", "key1")])
    storage.conn.execute("UPDATE chunks SET archived = 1 WHERE chunk_id = ?", ("chunk1",))
    storage.conn.commit()

    # Try to archive again
    stats = archive_by_age(storage, age_threshold_days=0)  # Archive all

    # Should not count already-archived chunk
    assert stats.archived_count == 0

    storage.close()
