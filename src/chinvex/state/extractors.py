# src/chinvex/state/extractors.py
import re
import logging
from datetime import datetime, timezone
from chinvex.state.models import RecentlyChanged, ExtractedTodo

log = logging.getLogger(__name__)


def extract_recently_changed(
    context: str,
    since: datetime,
    limit: int = 20,
    db_path: str = None
) -> list[RecentlyChanged]:
    """
    Get docs changed since last state generation.

    Args:
        context: Context name
        since: Only include docs changed after this time
        limit: Max number of results
        db_path: Override DB path (for testing)
    """
    # Import here to avoid circular dependency
    import sqlite3

    if db_path is None:
        db_path = f"P:/ai_memory/indexes/{context}/hybrid.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT source_uri, source_type, doc_id, last_ingested_at_unix
        FROM source_fingerprints
        WHERE context_name = ?
          AND last_ingested_at_unix > ?
          AND last_status = 'ok'
        ORDER BY last_ingested_at_unix DESC
        LIMIT ?
    """, [context, since.timestamp(), limit])

    results = []
    for row in cursor:
        results.append(RecentlyChanged(
            doc_id=row['doc_id'],
            source_type=row['source_type'],
            source_uri=row['source_uri'],
            change_type="modified",  # TODO: detect "new" vs "modified"
            changed_at=datetime.fromtimestamp(row['last_ingested_at_unix'], tz=timezone.utc)
        ))

    conn.close()
    return results


TODO_PATTERNS = [
    r"\bTODO[:\s](.+?)(?:\n|$)",        # Word boundary for TODO
    r"\bFIXME[:\s](.+?)(?:\n|$)",       # Word boundary for FIXME
    r"\bHACK[:\s](.+?)(?:\n|$)",        # Word boundary for HACK
    r"^\s*-?\s*\[\s\]\s+(.+)$",         # Checkbox at line start
    r"\bP[0-3][:\s](.+?)(?:\n|$)",      # P0, P1, P2, P3 with word boundary
]


def extract_todos(
    text: str,
    source_uri: str,
    doc_size: int | None = None
) -> list[ExtractedTodo]:
    """
    Extract TODO-like items from text.

    Args:
        text: Source text to scan
        source_uri: File/doc URI for attribution
        doc_size: Optional size check (skip huge files)

    Returns:
        List of extracted TODOs

    Note:
        Accepts false positives (TODOs in strings, not just comments).
        Skips files > 1MB to avoid performance issues.
    """
    # Safety: skip huge files
    if doc_size and doc_size > 1_000_000:
        log.debug(f"Skipping TODO extraction for {source_uri} (size={doc_size})")
        return []

    todos = []
    lines = text.split('\n')

    for i, line in enumerate(lines, 1):
        for pattern in TODO_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                todos.append(ExtractedTodo(
                    text=match.group(0).strip(),
                    source_uri=source_uri,
                    line=i,
                    extracted_at=datetime.now(timezone.utc)
                ))
                break  # One match per line

    return todos
