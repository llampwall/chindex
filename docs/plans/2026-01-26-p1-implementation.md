# P1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Chinvex from "working retrieval engine" to "daily-use memory system" with proactive awareness and real-time ingestion.

**Architecture:** Five parallel-capable components building on P0 foundation: (1) Real Codex app-server ingestion, (2) STATE.md deterministic state tracking, (3) Recency decay ranking, (4) Watch list for proactive alerts, (5) Optional LLM state consolidation.

**Tech Stack:** Python 3.12, SQLite FTS5, Chroma, Ollama, requests, Pydantic

---

## Phase 0: Foundation (Unblocks Everything)

### Task 1: Add IngestRunResult Dataclass

**Files:**
- Modify: `src/chinvex/ingest.py`
- Create: `tests/test_ingest_result.py`

**Step 1: Write the failing test**

```python
# tests/test_ingest_result.py
from datetime import datetime
from chinvex.ingest import IngestRunResult

def test_ingest_run_result_creation():
    """Test IngestRunResult can be created with all required fields."""
    result = IngestRunResult(
        run_id="test_run_123",
        context="TestContext",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        new_doc_ids=["doc1", "doc2"],
        updated_doc_ids=["doc3"],
        new_chunk_ids=["chunk1", "chunk2", "chunk3"],
        skipped_doc_ids=["doc4"],
        error_doc_ids=["doc5"],
        stats={"files_scanned": 10, "total_chunks": 25}
    )

    assert result.run_id == "test_run_123"
    assert result.context == "TestContext"
    assert len(result.new_doc_ids) == 2
    assert len(result.new_chunk_ids) == 3
    assert result.stats["files_scanned"] == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_result.py::test_ingest_run_result_creation -v`

Expected: FAIL with "cannot import name 'IngestRunResult'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/ingest.py (add at top after imports)
from dataclasses import dataclass
from datetime import datetime

@dataclass
class IngestRunResult:
    """Result of an ingest run, tracking what was processed."""
    run_id: str
    context: str
    started_at: datetime
    finished_at: datetime
    new_doc_ids: list[str]      # Docs ingested for first time
    updated_doc_ids: list[str]  # Docs re-ingested due to changes
    new_chunk_ids: list[str]    # All chunks created this run
    skipped_doc_ids: list[str]  # Docs skipped (unchanged)
    error_doc_ids: list[str]    # Docs that failed
    stats: dict                 # {files_scanned, total_chunks, etc.}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_result.py::test_ingest_run_result_creation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ingest_result.py src/chinvex/ingest.py
git commit -m "feat: add IngestRunResult dataclass for post-ingest hooks

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Refactor Ingest to Return IngestRunResult

**Files:**
- Modify: `src/chinvex/ingest.py`
- Modify: `tests/test_ingest_result.py`

**Step 1: Write the failing test**

```python
# tests/test_ingest_result.py (append)
def test_ingest_returns_result(tmp_path):
    """Test that ingest function returns IngestRunResult."""
    # This is a placeholder - adapt to your actual ingest signature
    from chinvex.ingest import ingest_all

    # Create minimal test context
    context = create_test_context(tmp_path)

    result = ingest_all(context)

    assert isinstance(result, IngestRunResult)
    assert result.run_id is not None
    assert result.context == context.name
    assert isinstance(result.new_chunk_ids, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_result.py::test_ingest_returns_result -v`

Expected: FAIL (ingest doesn't return IngestRunResult yet)

**Step 3: Refactor ingest to build and return IngestRunResult**

```python
# src/chinvex/ingest.py (modify main ingest function)
import uuid
from datetime import datetime, timezone

def ingest_all(context: Context) -> IngestRunResult:
    """Ingest all sources for a context, return detailed result."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc)

    new_doc_ids = []
    updated_doc_ids = []
    new_chunk_ids = []
    skipped_doc_ids = []
    error_doc_ids = []
    stats = {}

    # ... existing ingest logic, tracking each doc/chunk ...
    # When a doc is new: new_doc_ids.append(doc_id)
    # When a doc is updated: updated_doc_ids.append(doc_id)
    # When chunks created: new_chunk_ids.extend(chunk_ids)
    # When doc skipped: skipped_doc_ids.append(doc_id)
    # When doc errors: error_doc_ids.append(doc_id)

    finished_at = datetime.now(timezone.utc)

    return IngestRunResult(
        run_id=run_id,
        context=context.name,
        started_at=started_at,
        finished_at=finished_at,
        new_doc_ids=new_doc_ids,
        updated_doc_ids=updated_doc_ids,
        new_chunk_ids=new_chunk_ids,
        skipped_doc_ids=skipped_doc_ids,
        error_doc_ids=error_doc_ids,
        stats=stats
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_result.py::test_ingest_returns_result -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ingest_result.py src/chinvex/ingest.py
git commit -m "refactor: make ingest return IngestRunResult with tracking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 1: Parallel Development Tracks

### Track A: Real Codex App-Server Ingestion (P1.1)

#### Task 3: Create Codex App-Server Adapter Structure

**Files:**
- Create: `src/chinvex/adapters/__init__.py`
- Create: `src/chinvex/adapters/cx_appserver/__init__.py`
- Create: `src/chinvex/adapters/cx_appserver/client.py`
- Create: `src/chinvex/adapters/cx_appserver/schemas.py`
- Create: `tests/adapters/test_cx_client.py`

**Step 1: Write the failing test**

```python
# tests/adapters/test_cx_client.py
from chinvex.adapters.cx_appserver.client import CodexAppServerClient

def test_client_init():
    """Test CodexAppServerClient can be initialized."""
    client = CodexAppServerClient(base_url="http://localhost:8080")
    assert client.base_url == "http://localhost:8080"
    assert client.session is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_cx_client.py::test_client_init -v`

Expected: FAIL with "cannot import name 'CodexAppServerClient'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/adapters/__init__.py
"""Adapters for various data sources."""

# src/chinvex/adapters/cx_appserver/__init__.py
"""Codex app-server adapter."""
from .client import CodexAppServerClient

__all__ = ["CodexAppServerClient"]

# src/chinvex/adapters/cx_appserver/client.py
import requests

class CodexAppServerClient:
    """Client for Codex app-server API."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize client with base URL."""
        self.base_url = base_url
        self.session = requests.Session()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_cx_client.py::test_client_init -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/ tests/adapters/
git commit -m "feat: create Codex app-server adapter structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 4: Implement Health Check Method

**Files:**
- Modify: `src/chinvex/adapters/cx_appserver/client.py`
- Modify: `tests/adapters/test_cx_client.py`

**Step 1: Write the failing test**

```python
# tests/adapters/test_cx_client.py (append)
from unittest.mock import Mock, patch

def test_health_check_success():
    """Test health check returns True when server is reachable."""
    client = CodexAppServerClient()

    with patch.object(client.session, 'get') as mock_get:
        mock_get.return_value = Mock(status_code=200)
        success, msg = client.health_check()

    assert success is True
    assert "reachable" in msg.lower()

def test_health_check_connection_refused():
    """Test health check returns False on connection error."""
    client = CodexAppServerClient()

    with patch.object(client.session, 'get') as mock_get:
        mock_get.side_effect = requests.ConnectionError()
        success, msg = client.health_check()

    assert success is False
    assert "connection refused" in msg.lower()
    assert "localhost:8080" in msg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_cx_client.py::test_health_check_success -v`

Expected: FAIL with "'CodexAppServerClient' object has no attribute 'health_check'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/adapters/cx_appserver/client.py (add method)
def health_check(self) -> tuple[bool, str]:
    """
    Check if app-server is reachable.

    Returns:
        (success, message) tuple with detailed error info
    """
    try:
        resp = self.session.get(f"{self.base_url}/health", timeout=5)
        if resp.status_code == 401:
            return (False, "Authentication failed (401). Check godex credentials.")
        if resp.status_code == 200:
            return (True, "App-server reachable")
        return (False, f"Unexpected status: {resp.status_code}")
    except requests.ConnectionError:
        return (False, f"Connection refused. Is app-server running at {self.base_url}?")
    except requests.Timeout:
        return (False, "Timeout connecting to app-server")
    except Exception as e:
        return (False, f"Unexpected error: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_cx_client.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/cx_appserver/client.py tests/adapters/test_cx_client.py
git commit -m "feat: add health check to Codex app-server client

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 5: Define API Response Schemas

**Files:**
- Create: `src/chinvex/adapters/cx_appserver/schemas.py`
- Create: `tests/adapters/test_cx_schemas.py`

**Step 1: Write the failing test**

```python
# tests/adapters/test_cx_schemas.py
from datetime import datetime
from chinvex.adapters.cx_appserver.schemas import ThreadSummary, ThreadDetail, ThreadMessage

def test_thread_summary_creation():
    """Test ThreadSummary model validation."""
    summary = ThreadSummary(
        id="thread_123",
        title="Test Thread",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        message_count=5
    )
    assert summary.id == "thread_123"
    assert summary.title == "Test Thread"

def test_thread_message_creation():
    """Test ThreadMessage model validation."""
    msg = ThreadMessage(
        id="msg_123",
        role="user",
        content="Hello world",
        timestamp=datetime.now()
    )
    assert msg.role == "user"
    assert msg.content == "Hello world"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_cx_schemas.py::test_thread_summary_creation -v`

Expected: FAIL with "cannot import name 'ThreadSummary'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/adapters/cx_appserver/schemas.py
from datetime import datetime
from pydantic import BaseModel

class ThreadSummary(BaseModel):
    """Summary of a Codex thread from list endpoint."""
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int | None = None

class ThreadMessage(BaseModel):
    """Single message in a thread."""
    id: str
    role: str  # user|assistant|tool|system
    content: str | None
    timestamp: datetime
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None
    model: str | None = None

class ThreadDetail(BaseModel):
    """Full thread with all messages."""
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[ThreadMessage]
    workspace_id: str | None = None
    repo_paths: list[str] | None = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_cx_schemas.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/cx_appserver/schemas.py tests/adapters/test_cx_schemas.py
git commit -m "feat: define Pydantic schemas for Codex app-server API

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 6: Implement Sample Capture for Schema Validation

**Files:**
- Create: `src/chinvex/adapters/cx_appserver/capture.py`
- Create: `tests/adapters/test_capture.py`
- Create: `debug/appserver_samples/.gitkeep`

**Step 1: Write the failing test**

```python
# tests/adapters/test_capture.py
from pathlib import Path
import json
from chinvex.adapters.cx_appserver.capture import capture_sample

def test_capture_sample_writes_file(tmp_path):
    """Test that capture_sample writes JSON to correct path."""
    sample_data = {"id": "test_123", "title": "Test"}

    # Temporarily override debug path
    with patch('chinvex.adapters.cx_appserver.capture.SAMPLE_DIR', tmp_path):
        capture_sample("thread_resume", sample_data, "TestContext")

    files = list(tmp_path.glob("TestContext/thread_resume_*.json"))
    assert len(files) == 1

    with open(files[0]) as f:
        loaded = json.load(f)
    assert loaded["id"] == "test_123"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_capture.py::test_capture_sample_writes_file -v`

Expected: FAIL with "cannot import name 'capture_sample'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/adapters/cx_appserver/capture.py
import json
from pathlib import Path
from datetime import datetime

SAMPLE_DIR = Path("debug/appserver_samples")

def capture_sample(endpoint: str, response: dict, context_name: str):
    """
    Save raw API response for schema validation.

    Args:
        endpoint: API endpoint name (e.g., "thread_resume")
        response: Raw API response dict
        context_name: Context this sample belongs to
    """
    sample_dir = SAMPLE_DIR / context_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{endpoint.replace('/', '_')}_{timestamp}.json"

    with open(sample_dir / filename, "w") as f:
        json.dump(response, f, indent=2, default=str)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_capture.py::test_capture_sample_writes_file -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/cx_appserver/capture.py tests/adapters/test_capture.py debug/appserver_samples/.gitkeep
git commit -m "feat: add API response capture for schema validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Track B: STATE.md Deterministic Layer (P1.2)

#### Task 7: Upgrade Context Schema to v2 with Auto-Migration

**Files:**
- Modify: `src/chinvex/config.py`
- Create: `tests/test_context_migration.py`

**Step 1: Write the failing test**

```python
# tests/test_context_migration.py
import json
from pathlib import Path
from chinvex.config import load_context, Context

def test_context_v1_to_v2_migration(tmp_path):
    """Test auto-upgrade from schema v1 to v2."""
    # Create v1 context
    context_dir = tmp_path / "contexts" / "TestCtx"
    context_dir.mkdir(parents=True)

    v1_data = {
        "schema_version": 1,
        "name": "TestCtx",
        "includes": {"repos": []},
        "index": {
            "sqlite_path": "test.db",
            "chroma_dir": "chroma/"
        },
        "weights": {"repo": 1.0}
    }

    context_file = context_dir / "context.json"
    context_file.write_text(json.dumps(v1_data))

    # Load context (should auto-upgrade)
    context = load_context("TestCtx", base_dir=tmp_path / "contexts")

    # Verify v2 fields added
    assert context.schema_version == 2
    assert hasattr(context, 'codex_appserver')
    assert hasattr(context, 'ranking')
    assert context.codex_appserver.enabled is False
    assert context.ranking.recency_enabled is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_migration.py::test_context_v1_to_v2_migration -v`

Expected: FAIL (migration not implemented yet)

**Step 3: Implement auto-migration in load_context**

```python
# src/chinvex/config.py (modify load_context function)
import logging

log = logging.getLogger(__name__)

def load_context(name: str, base_dir: Path = None) -> Context:
    """
    Load context with auto-upgrade from v1 → v2.

    Args:
        name: Context name
        base_dir: Override default contexts directory (for testing)
    """
    if base_dir is None:
        base_dir = Path("P:/ai_memory/contexts")

    context_path = base_dir / name / "context.json"
    raw = json.loads(context_path.read_text())

    # Auto-upgrade v1 → v2
    if raw.get("schema_version", 1) == 1:
        log.info(f"Auto-upgrading {name} context from v1 to v2")
        raw["schema_version"] = 2

        # Add P1 fields with defaults
        raw.setdefault("codex_appserver", {
            "enabled": False,
            "base_url": "http://localhost:8080",
            "ingest_limit": 100,
            "timeout_sec": 30
        })
        raw.setdefault("ranking", {
            "recency_enabled": True,
            "recency_half_life_days": 90
        })
        raw.setdefault("state_llm", None)

        # Save upgraded version
        context_path.write_text(json.dumps(raw, indent=2))
        log.info(f"Saved upgraded context to {context_path}")

    return Context.model_validate(raw)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_migration.py::test_context_v1_to_v2_migration -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/config.py tests/test_context_migration.py
git commit -m "feat: add context schema v2 auto-migration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

#### Task 8: Define state.json Schema and Models

**Files:**
- Create: `src/chinvex/state/__init__.py`
- Create: `src/chinvex/state/models.py`
- Create: `tests/state/test_state_models.py`

**Step 1: Write the failing test**

```python
# tests/state/test_state_models.py
from datetime import datetime
from chinvex.state.models import StateJson, RecentlyChanged, ActiveThread

def test_state_json_creation():
    """Test StateJson model with all fields."""
    state = StateJson(
        schema_version=1,
        context="TestContext",
        generated_at=datetime.now(),
        last_ingest_run="run_abc123",
        generation_status="ok",
        generation_error=None,
        recently_changed=[],
        active_threads=[],
        extracted_todos=[],
        watch_hits=[],
        decisions=[],
        facts=[],
        annotations=[]
    )
    assert state.context == "TestContext"
    assert state.generation_status == "ok"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/state/test_state_models.py::test_state_json_creation -v`

Expected: FAIL with "cannot import name 'StateJson'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/state/__init__.py
"""State management for Chinvex."""
from .models import StateJson

__all__ = ["StateJson"]

# src/chinvex/state/models.py
from datetime import datetime
from pydantic import BaseModel

class RecentlyChanged(BaseModel):
    """Document that changed recently."""
    doc_id: str
    source_type: str
    source_uri: str
    change_type: str  # "new" or "modified"
    changed_at: datetime
    summary: str | None = None

class ActiveThread(BaseModel):
    """Active Codex session thread."""
    id: str
    title: str
    status: str
    last_activity: datetime
    source: str

class ExtractedTodo(BaseModel):
    """TODO extracted from source code."""
    text: str
    source_uri: str
    line: int
    extracted_at: datetime

class WatchHit(BaseModel):
    """Watch query that matched new content."""
    watch_id: str
    query: str
    hits: list[dict]
    triggered_at: datetime

class StateJson(BaseModel):
    """State file schema (state.json)."""
    schema_version: int
    context: str
    generated_at: datetime
    last_ingest_run: str
    generation_status: str  # "ok", "partial", "failed"
    generation_error: str | None

    recently_changed: list[RecentlyChanged]
    active_threads: list[ActiveThread]
    extracted_todos: list[ExtractedTodo]
    watch_hits: list[WatchHit]
    decisions: list[dict]  # From LLM consolidator (P1.5)
    facts: list[dict]      # From LLM consolidator (P1.5)
    annotations: list[dict]  # Human annotations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/state/test_state_models.py::test_state_json_creation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/state/ tests/state/
git commit -m "feat: define state.json schema and Pydantic models

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Track C: Recency Decay (P1.3)

#### Task 9: Implement Recency Factor Function

**Files:**
- Create: `src/chinvex/ranking.py`
- Create: `tests/test_ranking.py`

**Step 1: Write the failing test**

```python
# tests/test_ranking.py
from datetime import datetime, timedelta, timezone
from chinvex.ranking import recency_factor

def test_recency_factor_current_document():
    """Test recency factor is 1.0 for current documents."""
    now = datetime.now(timezone.utc)
    factor = recency_factor(now, half_life_days=90)
    assert factor == 1.0

def test_recency_factor_old_document():
    """Test recency factor decays for old documents."""
    now = datetime.now(timezone.utc)
    ninety_days_ago = now - timedelta(days=90)

    factor = recency_factor(ninety_days_ago, half_life_days=90)
    assert 0.49 < factor < 0.51  # Should be ~0.5

def test_recency_factor_very_old_document():
    """Test recency factor for very old documents."""
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)

    factor = recency_factor(one_year_ago, half_life_days=90)
    assert 0 < factor < 0.1  # Should be very small but not zero
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ranking.py::test_recency_factor_current_document -v`

Expected: FAIL with "cannot import name 'recency_factor'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/ranking.py
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

def recency_factor(updated_at: datetime, half_life_days: int = 90) -> float:
    """
    Exponential decay. Score halves every `half_life_days`.

    Args:
        updated_at: When document was last updated
        half_life_days: Days for score to decay by half

    Returns:
        Decay factor in (0, 1]

    Note:
        Uses UTC everywhere to avoid DST/timezone bugs.
    """
    now = datetime.now(timezone.utc)

    # Ensure updated_at is timezone-aware
    if updated_at.tzinfo is None:
        # Assume naive timestamps are UTC (with warning)
        updated_at = updated_at.replace(tzinfo=timezone.utc)
        log.warning(f"Naive timestamp encountered, assuming UTC: {updated_at}")

    age_days = (now - updated_at).days
    if age_days <= 0:
        return 1.0

    return 0.5 ** (age_days / half_life_days)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ranking.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/ranking.py tests/test_ranking.py
git commit -m "feat: implement exponential recency decay for ranking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Integration Tasks

These tasks integrate the parallel tracks. They require coordination between components.

### Task 10: Implement Recently Changed Extractor

**Files:**
- Create: `src/chinvex/state/extractors.py`
- Create: `tests/state/test_extractors.py`

**Step 1: Write the failing test**

```python
# tests/state/test_extractors.py
from datetime import datetime, timedelta, timezone
from chinvex.state.extractors import extract_recently_changed

def test_extract_recently_changed(test_db):
    """Test extracting recently changed documents."""
    # Insert test fingerprints
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)

    # This test requires actual DB setup - adapt to your schema
    changed = extract_recently_changed(
        context="TestContext",
        since=cutoff,
        limit=20
    )

    assert isinstance(changed, list)
    # Further assertions based on test data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/state/test_extractors.py::test_extract_recently_changed -v`

Expected: FAIL with "cannot import name 'extract_recently_changed'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/state/extractors.py
from datetime import datetime
from chinvex.state.models import RecentlyChanged

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
            changed_at=datetime.fromtimestamp(row['last_ingested_at_unix'])
        ))

    conn.close()
    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/state/test_extractors.py::test_extract_recently_changed -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/state/extractors.py tests/state/test_extractors.py
git commit -m "feat: implement recently_changed extractor for STATE.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 11: Implement TODO Extractor

**Files:**
- Modify: `src/chinvex/state/extractors.py`
- Modify: `tests/state/test_extractors.py`

**Step 1: Write the failing test**

```python
# tests/state/test_extractors.py (append)
from chinvex.state.extractors import extract_todos

def test_extract_todos_from_text():
    """Test TODO extraction with various patterns."""
    text = """
    # TODO: implement feature X
    def foo():
        pass  # FIXME: handle edge case

    - [ ] unchecked task
    P0: critical bug fix needed
    """

    todos = extract_todos(text, "test.py")

    assert len(todos) >= 4
    assert any("TODO" in t.text for t in todos)
    assert any("FIXME" in t.text for t in todos)
    assert any("[ ]" in t.text for t in todos)
    assert any("P0" in t.text for t in todos)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/state/test_extractors.py::test_extract_todos_from_text -v`

Expected: FAIL with "cannot import name 'extract_todos'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/state/extractors.py (append)
import re
import logging
from chinvex.state.models import ExtractedTodo

log = logging.getLogger(__name__)

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
                    extracted_at=datetime.now()
                ))
                break  # One match per line

    return todos
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/state/test_extractors.py::test_extract_todos_from_text -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/state/extractors.py tests/state/test_extractors.py
git commit -m "feat: implement TODO extraction with multiple patterns

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 12: Implement Active Threads Extractor

**Files:**
- Modify: `src/chinvex/state/extractors.py`
- Modify: `tests/state/test_extractors.py`

**Step 1: Write the failing test**

```python
# tests/state/test_extractors.py (append)
from chinvex.state.extractors import extract_active_threads

def test_extract_active_threads(test_db):
    """Test extracting active Codex session threads."""
    threads = extract_active_threads(
        context="TestContext",
        days=7,
        limit=20
    )

    assert isinstance(threads, list)
    # Empty list OK if no Codex ingestion yet
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/state/test_extractors.py::test_extract_active_threads -v`

Expected: FAIL with "cannot import name 'extract_active_threads'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/state/extractors.py (append)
from datetime import timedelta, timezone
from chinvex.state.models import ActiveThread

def extract_active_threads(
    context: str,
    days: int = 7,
    limit: int = 20,
    db_path: str = None
) -> list[ActiveThread]:
    """
    Codex sessions with activity in the last N days.

    Args:
        context: Context name
        days: Look back window
        limit: Max number of results
        db_path: Override DB path (for testing)

    Returns:
        List of active threads (empty if P1.1 not run yet)

    Note:
        Filter by source_type='codex_session' (the category).
        If P1.1 hasn't run yet, returns empty list (not an error).
    """
    import sqlite3

    if db_path is None:
        db_path = f"P:/ai_memory/indexes/{context}/hybrid.db"

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT doc_id, source_uri, title, updated_at
        FROM documents
        WHERE source_type = 'codex_session'
          AND updated_at > ?
        ORDER BY updated_at DESC
        LIMIT ?
    """, [cutoff.isoformat(), limit])

    results = []
    for row in cursor:
        results.append(ActiveThread(
            id=row['doc_id'],
            title=row['title'] or "Untitled",
            status="open",
            last_activity=datetime.fromisoformat(row['updated_at']),
            source="codex_session"
        ))

    conn.close()

    if not results:
        log.debug(f"No codex_session docs found (P1.1 not run yet?)")

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/state/test_extractors.py::test_extract_active_threads -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/state/extractors.py tests/state/test_extractors.py
git commit -m "feat: implement active threads extractor for Codex sessions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 13: Implement STATE.md Renderer

**Files:**
- Create: `src/chinvex/state/renderer.py`
- Create: `tests/state/test_renderer.py`

**Step 1: Write the failing test**

```python
# tests/state/test_renderer.py
from datetime import datetime
from chinvex.state.renderer import render_state_md
from chinvex.state.models import StateJson, RecentlyChanged

def test_render_state_md_basic():
    """Test STATE.md rendering with basic data."""
    state = StateJson(
        schema_version=1,
        context="TestContext",
        generated_at=datetime.now(),
        last_ingest_run="run_abc",
        generation_status="ok",
        generation_error=None,
        recently_changed=[
            RecentlyChanged(
                doc_id="doc1",
                source_type="repo",
                source_uri="C:\\test\\file.py",
                change_type="modified",
                changed_at=datetime.now()
            )
        ],
        active_threads=[],
        extracted_todos=[],
        watch_hits=[],
        decisions=[],
        facts=[],
        annotations=[]
    )

    md = render_state_md(state)

    assert "STATE.md - TestContext" in md
    assert "AUTOGENERATED" in md
    assert "Recently Changed" in md
    assert "file.py" in md
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/state/test_renderer.py::test_render_state_md_basic -v`

Expected: FAIL with "cannot import name 'render_state_md'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/state/renderer.py
from pathlib import Path
from chinvex.state.models import StateJson

def render_state_md(state: StateJson) -> str:
    """
    Render state.json to human-readable markdown.

    Args:
        state: StateJson object to render

    Returns:
        Markdown string for STATE.md
    """
    lines = [
        "<!-- AUTOGENERATED. Edits will be overwritten. Use 'chinvex state note add' for annotations. -->",
        "",
        f"# STATE.md - {state.context}",
        "",
        f"*Generated: {state.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    # Generation status warning
    if state.generation_status != "ok":
        lines.append(f"⚠️ **Generation Status:** {state.generation_status}")
        if state.generation_error:
            lines.append(f"*Error: {state.generation_error}*")
        lines.append("")

    # Recently Changed
    if state.recently_changed:
        lines.append("## Recently Changed")
        lines.append("")
        for item in state.recently_changed[:10]:
            name = Path(item.source_uri).name
            ts = item.changed_at.strftime('%Y-%m-%d %H:%M')
            lines.append(f"- `{name}` ({item.source_type}) - {ts}")
        lines.append("")

    # Active Threads
    if state.active_threads:
        lines.append("## Active Threads")
        lines.append("")
        for thread in state.active_threads:
            ts = thread.last_activity.strftime('%Y-%m-%d')
            lines.append(f"- **{thread.title}** - {ts}")
        lines.append("")

    # TODOs
    if state.extracted_todos:
        lines.append("## TODOs")
        lines.append("")
        for todo in state.extracted_todos[:20]:
            name = Path(todo.source_uri).name
            lines.append(f"- [ ] {todo.text} (`{name}:{todo.line}`)")
        lines.append("")

    # Watch Hits
    if state.watch_hits:
        lines.append("## Watch Hits (since last ingest)")
        lines.append("")
        for hit in state.watch_hits:
            lines.append(f"### {hit.watch_id}")
            for chunk in hit.hits[:3]:
                score = chunk.get('score', 0)
                snippet = chunk.get('snippet', '')[:100]
                lines.append(f"- [{score:.2f}] {snippet}...")
        lines.append("")

    # Decisions (if any, from P1.5)
    if state.decisions:
        lines.append("## Decisions")
        lines.append("")
        for dec in state.decisions:
            text = dec.get('text', '')
            date = dec.get('date', '')
            lines.append(f"- {text} ({date})")
        lines.append("")

    # Facts (if any, from P1.5)
    if state.facts:
        lines.append("## Facts")
        lines.append("")
        for fact in state.facts:
            text = fact.get('text', '')
            lines.append(f"- {text}")
        lines.append("")

    # Annotations
    if state.annotations:
        lines.append("## Annotations")
        lines.append("")
        for ann in state.annotations:
            text = ann.get('text', '')
            lines.append(f"- {text}")
        lines.append("")

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/state/test_renderer.py::test_render_state_md_basic -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/state/renderer.py tests/state/test_renderer.py
git commit -m "feat: implement STATE.md markdown renderer

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 14: Implement Watch System

**Files:**
- Create: `src/chinvex/watch/__init__.py`
- Create: `src/chinvex/watch/models.py`
- Create: `src/chinvex/watch/runner.py`
- Create: `tests/watch/test_watch.py`

**Step 1: Write the failing test**

```python
# tests/watch/test_watch.py
from chinvex.watch.models import Watch

def test_watch_creation():
    """Test Watch model creation."""
    watch = Watch(
        id="test_watch",
        query="test query",
        min_score=0.75,
        enabled=True,
        created_at="2026-01-26T00:00:00Z"
    )
    assert watch.id == "test_watch"
    assert watch.enabled is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/watch/test_watch.py::test_watch_creation -v`

Expected: FAIL with "cannot import name 'Watch'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/watch/__init__.py
"""Watch system for proactive alerts."""
from .models import Watch

__all__ = ["Watch"]

# src/chinvex/watch/models.py
from pydantic import BaseModel

class Watch(BaseModel):
    """Watch configuration for monitoring queries."""
    id: str
    query: str
    min_score: float
    enabled: bool
    created_at: str  # ISO8601

class WatchConfig(BaseModel):
    """watch.json file schema."""
    schema_version: int = 1
    watches: list[Watch]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/watch/test_watch.py::test_watch_creation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/watch/ tests/watch/
git commit -m "feat: define watch system models and schema

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 15: Implement Watch Runner

**Files:**
- Modify: `src/chinvex/watch/runner.py`
- Modify: `tests/watch/test_watch.py`

**Step 1: Write the failing test**

```python
# tests/watch/test_watch.py (append)
from chinvex.watch.runner import run_watches
from chinvex.watch.models import Watch

def test_run_watches_basic(test_context):
    """Test watch execution returns hits."""
    watches = [
        Watch(
            id="test",
            query="test query",
            min_score=0.5,
            enabled=True,
            created_at="2026-01-26T00:00:00Z"
        )
    ]

    # Mock new chunk IDs
    new_chunk_ids = ["chunk1", "chunk2"]

    hits = run_watches(test_context, new_chunk_ids, watches)

    assert isinstance(hits, list)
    # Further assertions depend on mock data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/watch/test_watch.py::test_run_watches_basic -v`

Expected: FAIL with "cannot import name 'run_watches'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/watch/runner.py
import logging
from datetime import datetime
from chinvex.state.models import WatchHit

log = logging.getLogger(__name__)

def run_watches(
    context,
    new_chunk_ids: list[str],
    watches: list,
    timeout_per_watch: int = 30
) -> list[WatchHit]:
    """
    Run all enabled watches against newly ingested chunks.

    Args:
        context: Context object with search capability
        new_chunk_ids: List of newly created chunk IDs
        watches: List of Watch objects
        timeout_per_watch: Timeout in seconds per watch

    Returns:
        List of WatchHit objects for matches

    Note:
        Timeouts and errors are logged but don't fail entire run.
    """
    hits = []

    for watch in watches:
        if not watch.enabled:
            continue

        try:
            # Import search function (avoid circular import)
            from chinvex.search import search_chunks

            # Search only new chunks
            results = search_chunks(
                context=context,
                query=watch.query,
                chunk_ids=new_chunk_ids,
                k=10
            )

            # Filter by min_score
            matching = [r for r in results if r.blended_score >= watch.min_score]

            if matching:
                hits.append(WatchHit(
                    watch_id=watch.id,
                    query=watch.query,
                    hits=[
                        {
                            "chunk_id": r.chunk_id,
                            "score": r.blended_score,
                            "snippet": r.text[:200]
                        }
                        for r in matching[:5]
                    ],
                    triggered_at=datetime.now()
                ))

        except TimeoutError:
            log.warning(f"Watch {watch.id} timed out after {timeout_per_watch}s, skipping")
            continue
        except Exception as e:
            log.error(f"Watch {watch.id} failed: {e}")
            continue

    return hits
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/watch/test_watch.py::test_run_watches_basic -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/watch/runner.py tests/watch/test_watch.py
git commit -m "feat: implement watch runner with timeout and error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 16: Implement Post-Ingest Hook

**Files:**
- Create: `src/chinvex/hooks.py`
- Create: `tests/test_hooks.py`
- Modify: `src/chinvex/ingest.py`

**Step 1: Write the failing test**

```python
# tests/test_hooks.py
from chinvex.hooks import post_ingest_hook
from chinvex.ingest import IngestRunResult
from datetime import datetime

def test_post_ingest_hook_success(test_context, tmp_path):
    """Test post-ingest hook generates state."""
    result = IngestRunResult(
        run_id="test_run",
        context="TestContext",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        new_doc_ids=[],
        updated_doc_ids=[],
        new_chunk_ids=["chunk1"],
        skipped_doc_ids=[],
        error_doc_ids=[],
        stats={}
    )

    # Should not raise
    post_ingest_hook(test_context, result)

    # Verify state.json created
    state_path = tmp_path / "state.json"
    assert state_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hooks.py::test_post_ingest_hook_success -v`

Expected: FAIL with "cannot import name 'post_ingest_hook'"

**Step 3: Write minimal implementation**

```python
# src/chinvex/hooks.py
import json
import logging
from pathlib import Path
from datetime import datetime
from chinvex.state.models import StateJson
from chinvex.state.extractors import extract_recently_changed, extract_active_threads, extract_todos
from chinvex.state.renderer import render_state_md

log = logging.getLogger(__name__)

def post_ingest_hook(context, result):
    """
    Called after every ingest run to generate STATE.md.

    Args:
        context: Context object
        result: IngestRunResult from ingest

    Note:
        State generation failures DO NOT fail ingest (best-effort).
    """
    try:
        # Extract state components
        since = result.started_at

        recently_changed = extract_recently_changed(
            context=context.name,
            since=since,
            limit=20
        )

        active_threads = extract_active_threads(
            context=context.name,
            days=7,
            limit=20
        )

        # Extract TODOs from recently changed files
        todos = []
        # TODO: implement TODO extraction from changed files

        # Run watches (P1.4)
        watch_hits = []
        # TODO: load watches and run them

        # Create state
        state = StateJson(
            schema_version=1,
            context=context.name,
            generated_at=datetime.now(),
            last_ingest_run=result.run_id,
            generation_status="ok",
            generation_error=None,
            recently_changed=recently_changed,
            active_threads=active_threads,
            extracted_todos=todos,
            watch_hits=watch_hits,
            decisions=[],
            facts=[],
            annotations=[]
        )

    except Exception as e:
        log.error(f"State generation failed: {e}")
        # Create minimal error state
        state = StateJson(
            schema_version=1,
            context=context.name,
            generated_at=datetime.now(),
            last_ingest_run=result.run_id,
            generation_status="failed",
            generation_error=str(e),
            recently_changed=[],
            active_threads=[],
            extracted_todos=[],
            watch_hits=[],
            decisions=[],
            facts=[],
            annotations=[]
        )

    # Save state.json
    state_path = Path(f"P:/ai_memory/contexts/{context.name}/state.json")
    state_path.write_text(state.model_dump_json(indent=2))

    # Render and save STATE.md
    md = render_state_md(state)
    md_path = Path(f"P:/ai_memory/contexts/{context.name}/STATE.md")
    md_path.write_text(md)

    if state.generation_status == "ok":
        log.info(f"STATE.md updated: {len(result.new_chunk_ids)} new chunks")
    else:
        log.warning(f"STATE.md updated with errors: {state.generation_error}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hooks.py::test_post_ingest_hook_success -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/hooks.py tests/test_hooks.py
git commit -m "feat: implement post-ingest hook for state generation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 17: Wire Post-Ingest Hook to Ingest

**Files:**
- Modify: `src/chinvex/ingest.py`

**Step 1: Write the failing test**

```python
# tests/test_ingest_result.py (append)
def test_ingest_calls_post_hook(test_context, tmp_path, monkeypatch):
    """Test that ingest calls post-ingest hook."""
    hook_called = []

    def mock_hook(context, result):
        hook_called.append((context, result))

    monkeypatch.setattr('chinvex.ingest.post_ingest_hook', mock_hook)

    result = ingest_all(test_context)

    assert len(hook_called) == 1
    assert hook_called[0][1] == result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_result.py::test_ingest_calls_post_hook -v`

Expected: FAIL (hook not called yet)

**Step 3: Wire hook into ingest**

```python
# src/chinvex/ingest.py (modify ingest_all function)
from chinvex.hooks import post_ingest_hook

def ingest_all(context: Context) -> IngestRunResult:
    """Ingest all sources for a context, return detailed result."""
    # ... existing ingest logic ...

    result = IngestRunResult(
        run_id=run_id,
        context=context.name,
        started_at=started_at,
        finished_at=finished_at,
        new_doc_ids=new_doc_ids,
        updated_doc_ids=updated_doc_ids,
        new_chunk_ids=new_chunk_ids,
        skipped_doc_ids=skipped_doc_ids,
        error_doc_ids=error_doc_ids,
        stats=stats
    )

    # Call post-ingest hook
    try:
        post_ingest_hook(context, result)
    except Exception as e:
        log.error(f"Post-ingest hook failed: {e}")
        # Don't fail ingest on hook failure

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_result.py::test_ingest_calls_post_hook -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/ingest.py tests/test_ingest_result.py
git commit -m "feat: wire post-ingest hook into ingest pipeline

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 18: Add CLI Commands for State Management

**Files:**
- Modify: `src/chinvex/cli.py`
- Create: `tests/test_cli_state.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_state.py
from click.testing import CliRunner
from chinvex.cli import cli

def test_state_generate_command():
    """Test chinvex state generate command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['state', 'generate', '--context', 'TestContext'])

    assert result.exit_code == 0
    assert "STATE.md" in result.output or "generated" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_state.py::test_state_generate_command -v`

Expected: FAIL with "no such command 'state'"

**Step 3: Add state CLI commands**

```python
# src/chinvex/cli.py (add new command group)
import click

@cli.group()
def state():
    """Manage context state and STATE.md."""
    pass

@state.command('generate')
@click.option('--context', required=True, help='Context name')
@click.option('--llm', is_flag=True, help='Enable LLM consolidation (P1.5)')
@click.option('--since', default='24h', help='Time window (e.g., 24h, 7d)')
def state_generate(context, llm, since):
    """Generate state.json and STATE.md."""
    from chinvex.config import load_context
    from chinvex.hooks import post_ingest_hook
    from chinvex.ingest import IngestRunResult
    from datetime import datetime, timedelta

    ctx = load_context(context)

    # Parse since duration
    # TODO: implement duration parsing

    # Create fake result for manual generation
    result = IngestRunResult(
        run_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        context=context,
        started_at=datetime.now(),
        finished_at=datetime.now(),
        new_doc_ids=[],
        updated_doc_ids=[],
        new_chunk_ids=[],
        skipped_doc_ids=[],
        error_doc_ids=[],
        stats={}
    )

    post_ingest_hook(ctx, result)
    click.echo(f"Generated STATE.md for {context}")

@state.command('show')
@click.option('--context', required=True, help='Context name')
def state_show(context):
    """Print STATE.md to stdout."""
    from pathlib import Path

    md_path = Path(f"P:/ai_memory/contexts/{context}/STATE.md")
    if not md_path.exists():
        click.echo(f"No STATE.md found for {context}", err=True)
        return

    click.echo(md_path.read_text())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_state.py::test_state_generate_command -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/cli.py tests/test_cli_state.py
git commit -m "feat: add state CLI commands (generate, show)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan implements P1 in 18 bite-sized tasks across 3 parallel tracks:

**Phase 0 (Foundation):**
- Tasks 1-2: IngestRunResult infrastructure

**Phase 1 (Parallel Tracks):**
- Track A (P1.1): Tasks 3-6: Codex app-server ingestion foundation
- Track B (P1.2): Tasks 7-8: STATE.md schema and models
- Track C (P1.3): Task 9: Recency decay

**Phase 2 (Integration):**
- Tasks 10-13: State extractors and renderer
- Tasks 14-15: Watch system
- Tasks 16-17: Post-ingest hook integration
- Task 18: CLI commands

**Remaining Work (Not in this plan):**
- Complete Codex client implementation (list_threads, get_thread, auth)
- Wire Codex ingestion into pipeline
- Add watch CLI commands (add, list, remove, dry-run)
- Integrate recency decay into search ranking
- P1.5 LLM consolidator (optional)

Each task follows TDD with test-first development and includes a git commit step.

---

Plan complete and saved to `docs/plans/2026-01-26-p1-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
