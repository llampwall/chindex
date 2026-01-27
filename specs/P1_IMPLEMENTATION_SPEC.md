# Chinvex P1 Implementation Spec

**Version:** 1.1
**Date:** 2026-01-26
**Status:** Ready for implementation
**Depends on:** P0 complete (with IngestRunResult added)

---

## Overview

P1 turns Chinvex from "working retrieval engine" into "daily-use memory system." The core unlock is making it feel alive—it knows what changed, what's current, and can tap you on the shoulder.

### P1 Scope

1. **P1.1** Real Codex app-server ingestion (unmock it)
2. **P1.2** STATE.md deterministic layer (no LLM required)
3. **P1.3** Recency decay in ranking
4. **P1.4** Watch list (proactive alerts)
5. **P1.5** STATE.md LLM consolidator (optional enhancement)

### Non-Goals (P2+)

- Hosted HTTPS endpoint for ChatGPT Actions
- Cross-context search
- Claim-level grounding (sentence-level)
- Multi-user support
- Fancy UI

---

## 1. Real Codex App-Server Ingestion (P1.1)

### Current State

P0 smoke test uses a mocked app-server. The adapter structure exists but isn't wired to real endpoints.

### Goal

Ingest actual Codex threads from the app-server API, end-to-end.

### Done Criteria

P1.1 is complete when:
- It runs against your local app-server (not mocked)
- Captured samples are committed under `debug/appserver_samples/`
- Schemas are validated against real samples
- Threads appear in search results

### Source: Godex

Port the working client code from godex. It already handles:
- Authentication/session management
- `thread/list` enumeration
- `thread/resume` content fetching

### Implementation

#### File Structure

```
src/chinvex/adapters/cx_appserver/
├── client.py        # HTTP client (port from godex)
├── schemas.py       # Pydantic models for API responses
├── normalize.py     # Convert to ConversationDoc
├── capture.py       # Debug sample capture
└── __init__.py
```

#### Client Contract

```python
# client.py

class CodexAppServerClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize client.
        base_url: Local app-server endpoint (not cloud).
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def list_threads(
        self,
        limit: int = 50,
        offset: int = 0,
        after: datetime | None = None
    ) -> list[ThreadSummary]:
        """
        Enumerate threads, newest first.
        If `after` provided, only return threads updated after that time.
        """
        ...
    
    def get_thread(self, thread_id: str) -> ThreadDetail:
        """
        Fetch full thread content including all turns.
        """
        ...
    
    def health_check(self) -> bool:
        """
        Return True if app-server is reachable.
        """
        ...

    @classmethod
    def from_godex_env(cls) -> "CodexAppServerClient":
        """
        Load authentication from godex environment.
        Reuses godex's session/token management.

        Raises:
            EnvironmentError: If godex not configured
        """
        ...
```

#### Authentication Strategy

**Port from godex verbatim.** Godex already handles authentication (cookies/tokens/sessions). Don't reinvent:

```python
# Import godex's auth module and reuse its session
from godex.auth import get_authenticated_session

class CodexAppServerClient:
    @classmethod
    def from_godex_env(cls) -> "CodexAppServerClient":
        session = get_authenticated_session()
        client = cls()
        client.session = session
        return client
```

**Error messages must be specific:**
```python
def health_check(self) -> tuple[bool, str]:
    """Returns (success, message)"""
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
```

#### Schema Capture (Required)

Before finalizing schemas, capture real responses:

```python
# capture.py

def capture_sample(endpoint: str, response: dict, context_name: str):
    """
    Save raw API response for schema validation.
    """
    sample_dir = Path(f"debug/appserver_samples/{context_name}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{endpoint.replace('/', '_')}_{timestamp}.json"
    
    with open(sample_dir / filename, "w") as f:
        json.dump(response, f, indent=2, default=str)
```

**Run capture in dev, then lock schemas from real data.**

#### Schemas (Validate Against Samples)

```python
# schemas.py
from pydantic import BaseModel

class ThreadSummary(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int | None = None

class ThreadMessage(BaseModel):
    id: str
    role: str  # user|assistant|tool|system
    content: str | None
    timestamp: datetime
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None
    model: str | None = None

class ThreadDetail(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[ThreadMessage]
    workspace_id: str | None = None
    repo_paths: list[str] | None = None
```

**These are starting points. Update from captured samples.**

#### Normalization

```python
# normalize.py

def normalize_thread(thread: ThreadDetail) -> ConversationDoc:
    """
    Convert app-server ThreadDetail to internal ConversationDoc.
    """
    return ConversationDoc(
        doc_type="conversation",
        source="cx_appserver",
        thread_id=thread.id,
        title=thread.title or _extract_title(thread.messages),
        created_at=thread.created_at.isoformat(),
        updated_at=thread.updated_at.isoformat(),
        turns=[
            Turn(
                turn_id=msg.id,
                ts=msg.timestamp.isoformat(),
                role=msg.role,
                text=msg.content or "",
                tool=_extract_tool(msg),
                meta={"model": msg.model} if msg.model else {}
            )
            for msg in thread.messages
        ],
        links=Links(
            workspace_id=thread.workspace_id,
            repo_paths=thread.repo_paths or []
        )
    )

def _extract_title(messages: list[ThreadMessage]) -> str:
    """First user message, truncated."""
    for msg in messages:
        if msg.role == "user" and msg.content:
            return msg.content[:80] + ("..." if len(msg.content) > 80 else "")
    return "Untitled"

def _extract_tool(msg: ThreadMessage) -> dict | None:
    if msg.tool_calls:
        return {"calls": msg.tool_calls, "results": msg.tool_results}
    return None
```

#### Error Handling Strategy

**Best-effort per thread, not all-or-nothing:**

| Scenario | Behavior |
|----------|----------|
| App-server down at start | Fail fast with clear message |
| `list_threads` succeeds, `get_thread` fails | Log error, record fingerprint error, continue to next |
| Network timeout | Retry 2x with exponential backoff (1s, 2s) |
| Malformed thread content | Record `last_status='error'` in fingerprint, skip |

Ingest exit code = 0 if any threads succeeded.

#### Pagination Strategy

Default: Fetch `ingest_limit` threads (default 100).

```python
def ingest_codex_sessions(context: Context, full: bool = False, since: datetime | None = None):
    """
    Ingest Codex threads from app-server.

    Args:
        context: Context with codex_appserver config
        full: If True, paginate until exhausted. If False, respect ingest_limit.
        since: If provided, stop when thread.updated_at < since (early stop).
    """
    client = CodexAppServerClient.from_godex_env()

    ok, msg = client.health_check()
    if not ok:
        raise RuntimeError(f"Codex app-server not reachable: {msg}")

    limit = context.codex_appserver.ingest_limit
    offset = 0
    ingested_count = 0
    error_count = 0
    
    while True:
        threads = client.list_threads(limit=limit, offset=offset)
        if not threads:
            break  # No more threads

        for summary in threads:
            # Early stop if since cutoff reached
            if since and summary.updated_at < since:
                log.info(f"Reached cutoff ({since}), stopping pagination")
                return ingested_count, error_count

            # Check fingerprint
            should_ingest, reason = should_ingest_thread(
                thread_id=summary.id,
                context=context.name,
                current_updated_at=summary.updated_at.isoformat()
            )

            if not should_ingest and not full:
                log.debug(f"Skipping {summary.id}: {reason}")
                continue

            # Fetch full thread with retry
            try:
                thread = fetch_thread_with_retry(client, summary.id, retries=2)
            except Exception as e:
                log.error(f"Failed to fetch {summary.id}: {e}")
                update_thread_fingerprint(
                    thread_id=summary.id,
                    context=context.name,
                    last_status="error",
                    last_error=str(e)
                )
                error_count += 1
                continue  # Skip this thread, continue to next

            # Capture sample in dev mode
            if settings.dev_mode:
                capture_sample("thread_resume", thread.model_dump(), context.name)

            # Normalize
            try:
                doc = normalize_thread(thread)
            except Exception as e:
                log.error(f"Failed to normalize {summary.id}: {e}")
                error_count += 1
                continue

            # Chunk
            chunks = chunk_conversation(doc)

            # Embed + store
            embed_and_store(context, doc, chunks)

            # Update fingerprint (use updated_at only, not last_turn_id)
            update_thread_fingerprint(
                thread_id=summary.id,
                context=context.name,
                updated_at=summary.updated_at.isoformat(),
                last_status="ok"
            )

            ingested_count += 1

        # Pagination control
        if not full:
            break  # Single page only
        offset += limit

    log.info(f"Ingested {ingested_count} threads, {error_count} errors")
    return ingested_count, error_count


def fetch_thread_with_retry(client: CodexAppServerClient, thread_id: str, retries: int = 2) -> ThreadDetail:
    """Fetch thread with exponential backoff retry."""
    for attempt in range(retries + 1):
        try:
            return client.get_thread(thread_id)
        except requests.Timeout:
            if attempt < retries:
                wait = 2 ** attempt  # 1s, 2s
                log.warning(f"Timeout fetching {thread_id}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def should_ingest_thread(thread_id: str, context: str, current_updated_at: str) -> tuple[bool, str]:
    """
    Check if thread should be re-ingested based on fingerprint.

    Uses ONLY updated_at for change detection (not last_turn_id).
    If app-server reports updated_at change but content is identical, harmless re-ingest.
    """
    fp = get_fingerprint(thread_id, context)

    if fp is None:
        return (True, "new_thread")

    if fp.last_status == "error":
        return (True, "retry_after_error")

    if fp.thread_updated_at != current_updated_at:
        return (True, "thread_updated")

    if fp.parser_version != CURRENT_PARSER_VERSION:
        return (True, "parser_upgraded")

    if fp.chunker_version != CURRENT_CHUNKER_VERSION:
        return (True, "chunker_upgraded")

    if fp.embedded_model != CURRENT_EMBED_MODEL:
        return (True, "embed_model_changed")

    return (False, "unchanged")
```

#### Context Configuration (Schema v2)

P1 upgrades `context.json` to **schema_version: 2** with auto-migration:

```json
{
  "schema_version": 2,
  "name": "Chinvex",
  "aliases": ["chindex", "chinvex-engine"],
  "includes": {
    "repos": ["C:\\Code\\chinvex"],
    "chat_roots": ["P:\\ai_memory\\projects\\Chinvex\\chats"],
    "codex_session_roots": ["C:\\Users\\Jordan\\.codex\\sessions"],
    "note_roots": []
  },
  "index": {
    "sqlite_path": "P:\\ai_memory\\indexes\\Chinvex\\hybrid.db",
    "chroma_dir": "P:\\ai_memory\\indexes\\Chinvex\\chroma"
  },
  "weights": {
    "repo": 1.0,
    "chat": 0.8,
    "codex_session": 0.9,
    "note": 0.7
  },
  "codex_appserver": {
    "enabled": false,
    "base_url": "http://localhost:8080",
    "ingest_limit": 100,
    "timeout_sec": 30
  },
  "ranking": {
    "recency_enabled": true,
    "recency_half_life_days": 90
  },
  "state_llm": null,
  "created_at": "2026-01-26T00:00:00Z",
  "updated_at": "2026-01-26T00:00:00Z"
}
```

**Auto-upgrade on load (context.py):**

```python
def load_context(name: str) -> Context:
    """Load context with auto-upgrade from v1 → v2."""
    context_path = Path(f"P:/ai_memory/contexts/{name}/context.json")
    raw = json.loads(context_path.read_text())

    if raw.get("schema_version", 1) == 1:
        # Upgrade v1 → v2
        raw["schema_version"] = 2
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
        log.info(f"Auto-upgraded {name} context to schema v2")

    return Context.model_validate(raw)
```

### Acceptance Tests

```bash
# Test 1: App-server health check
chinvex codex health --context Chinvex
# Expected: "App-server reachable at http://localhost:8080"

# Test 2: List threads
chinvex codex list --context Chinvex --limit 5
# Expected: Table of thread summaries with IDs, titles, dates

# Test 3: Ingest real threads
chinvex ingest --context Chinvex --source codex
# Expected: Threads ingested, fingerprints recorded

# Test 4: Search finds Codex content
chinvex search --context Chinvex "that bug we debugged yesterday"
# Expected: Results from Codex sessions appear

# Test 5: Re-ingest skips unchanged
chinvex ingest --context Chinvex --source codex
# Expected: "Skipped N threads (unchanged)"
```

---

## 2. STATE.md Deterministic Layer (P1.2)

### Purpose

STATE.md is the "current truth" file—what's active, what changed, what decisions hold. Unlike search (which answers "what did we discuss?"), STATE.md answers "what's true now?"

### Architecture

```
state.json (structured, machine-written)
    ↓ render
STATE.md (human-readable, optionally human-editable)
```

**Key insight:** Keep structured data in JSON, render to markdown. Don't parse markdown back—that's merge hell.

### Human Edit Policy

**STATE.md is regenerated on every ingest. Direct edits will be lost.**

Human annotations go through `state.json` via CLI:

```bash
# Add a manual note/annotation
chinvex state note add --context Chinvex "Decided to use Postgres instead of SQLite for prod"

# List annotations
chinvex state note list --context Chinvex

# Remove annotation
chinvex state note remove --context Chinvex --id 3
```

Annotations are stored in `state.json` under `annotations` and rendered into STATE.md under a dedicated section. This way:
- Machine-generated content regenerates freely
- Human annotations persist across regenerations
- No markdown parsing required

### state.json Schema

```json
{
  "schema_version": 1,
  "context": "Chinvex",
  "generated_at": "2026-01-26T15:00:00Z",
  "last_ingest_run": "run_abc123",
  "generation_status": "ok",
  "generation_error": null,

  "recently_changed": [
    {
      "doc_id": "abc123",
      "source_type": "repo",
      "source_uri": "C:\\Code\\chinvex\\src\\search.py",
      "change_type": "modified",
      "changed_at": "2026-01-26T14:30:00Z",
      "summary": null
    }
  ],
  
  "active_threads": [
    {
      "id": "thread_xyz",
      "title": "P2 HTTPS endpoint design",
      "status": "open",
      "last_activity": "2026-01-26T12:00:00Z",
      "source": "codex_session"
    }
  ],
  
  "extracted_todos": [
    {
      "text": "TODO: add quota rerank",
      "source_uri": "C:\\Code\\chinvex\\src\\search.py",
      "line": 142,
      "extracted_at": "2026-01-26T14:30:00Z"
    }
  ],
  
  "watch_hits": [
    {
      "watch_id": "p2_https",
      "query": "P2 HTTPS endpoint",
      "hits": [
        {"chunk_id": "def456", "score": 0.85, "snippet": "..."}
      ],
      "triggered_at": "2026-01-26T14:30:00Z"
    }
  ],
  
  "decisions": [],
  "facts": [],
  "annotations": []
}
```

### Field Definitions

| Section | Source | Update Trigger | Default Cap |
|---------|--------|----------------|-------------|
| `recently_changed` | Fingerprint diffs | Every ingest | 20 items |
| `active_threads` | Codex sessions with recent activity | Codex ingest | 20 items |
| `extracted_todos` | Regex scan of ingested files | Every ingest | 50 items |
| `watch_hits` | Watch queries on new chunks | Every ingest | Ephemeral (overwritten) |
| `decisions` | LLM consolidator (P1.5) | Optional | No cap |
| `facts` | LLM consolidator (P1.5) | Optional | No cap |
| `annotations` | Human edits | Manual | No cap |
| `generation_status` | State generation result | Every ingest | `"ok"`, `"partial"`, or `"failed"` |
| `generation_error` | Error message if failed | Every ingest | `null` if successful |

**Default time window:** Since `state.json.generated_at` (if missing, last 24 hours).

### Deterministic Extraction (No LLM)

#### Recently Changed

```python
def extract_recently_changed(
    context: str,
    since: datetime,
    limit: int = 20
) -> list[RecentlyChanged]:
    """
    Get docs changed since last state generation.
    """
    fps = db.execute("""
        SELECT source_uri, source_type, doc_id, last_ingested_at_unix
        FROM source_fingerprints
        WHERE context_name = ?
          AND last_ingested_at_unix > ?
          AND last_status = 'ok'
        ORDER BY last_ingested_at_unix DESC
        LIMIT ?
    """, [context, since.timestamp(), limit])
    
    return [
        RecentlyChanged(
            doc_id=row.doc_id,
            source_type=row.source_type,
            source_uri=row.source_uri,
            change_type="modified",  # or "new" if first ingest
            changed_at=datetime.fromtimestamp(row.last_ingested_at_unix)
        )
        for row in fps
    ]
```

#### Extracted TODOs

```python
TODO_PATTERNS = [
    r"\bTODO[:\s](.+?)(?:\n|$)",        # Word boundary for TODO
    r"\bFIXME[:\s](.+?)(?:\n|$)",       # Word boundary for FIXME
    r"\bHACK[:\s](.+?)(?:\n|$)",        # Word boundary for HACK
    r"^\s*-?\s*\[\s\]\s+(.+)$",         # Checkbox at line start
    r"\bP[0-3][:\s](.+?)(?:\n|$)",      # P0, P1, P2, P3 with word boundary
]

def extract_todos(text: str, source_uri: str, doc_size: int | None = None) -> list[ExtractedTodo]:
    """
    Extract TODO-like items from text.

    Skips files > 1MB to avoid performance issues.
    Accepts false positives (TODOs in strings, not just comments).
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

#### Active Threads

```python
def extract_active_threads(
    context: str,
    days: int = 7,
    limit: int = 20
) -> list[ActiveThread]:
    """
    Codex sessions with activity in the last N days.

    Note: Filter by source_type='codex_session' (the category).
    If P1.1 hasn't run yet, this returns empty list (not an error).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Query documents table where source_type='codex_session'
    docs = db.execute("""
        SELECT doc_id, source_uri, title, updated_at
        FROM documents
        WHERE source_type = 'codex_session'
          AND updated_at > ?
        ORDER BY updated_at DESC
        LIMIT ?
    """, [cutoff.isoformat(), limit])

    if not docs:
        log.debug(f"No codex_session docs found (P1.1 not run yet?)")
        return []  # Empty list, not error

    return [
        ActiveThread(
            id=row.doc_id,
            title=row.title,
            status="open",
            last_activity=row.updated_at,
            source="codex_session"
        )
        for row in docs
    ]
```

**Naming convention (lock this in):**
| Field | Meaning | Values |
|-------|---------|--------|
| `source_type` | Category of content | `repo`, `chat`, `codex_session`, `note` |
| `source` | Adapter that produced it | `cx_appserver`, `cx_sessions_jsonl`, `chatgpt_export`, etc. |
| `doc_type` | Document structure | `file`, `conversation`, `note` |

### STATE.md Rendering

```python
def render_state_md(state: StateJson) -> str:
    """
    Render state.json to human-readable markdown.
    """
    lines = [
        "<!-- AUTOGENERATED. Edits will be overwritten. Use 'chinvex state note add' for annotations. -->",
        "",
        f"# STATE.md - {state.context}",
        "",
        f"*Generated: {state.generated_at}*",
        "",
    ]

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
            lines.append(f"- `{name}` ({item.source_type}) - {item.changed_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
    
    # Active Threads
    if state.active_threads:
        lines.append("## Active Threads")
        lines.append("")
        for thread in state.active_threads:
            lines.append(f"- **{thread.title or 'Untitled'}** - {thread.last_activity.strftime('%Y-%m-%d')}")
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
                lines.append(f"- [{chunk.score:.2f}] {chunk.snippet[:100]}...")
        lines.append("")
    
    # Decisions (if any, from P1.5)
    if state.decisions:
        lines.append("## Decisions")
        lines.append("")
        for dec in state.decisions:
            lines.append(f"- {dec.text} ({dec.date})")
        lines.append("")
    
    # Facts (if any, from P1.5)
    if state.facts:
        lines.append("## Facts")
        lines.append("")
        for fact in state.facts:
            lines.append(f"- {fact.text}")
        lines.append("")
    
    return "\n".join(lines)
```

### File Locations

```
P:\ai_memory\contexts\<Context>\
├── context.json
├── state.json         # Structured state (machine-written)
├── STATE.md           # Rendered view (human-readable)
└── watch_history.jsonl  # Append-only watch hit log
```

### Watch History Log

Watch hits in `state.json` are ephemeral (overwritten each ingest). For persistence, watch hits are also appended to `watch_history.jsonl`:

```jsonl
{"timestamp": "2026-01-26T14:30:00Z", "watch_id": "p2_https", "query": "P2 HTTPS", "ingest_run_id": "run_abc123", "hits": [{"chunk_id": "def456", "score": 0.85, "snippet": "..."}]}
{"timestamp": "2026-01-26T15:00:00Z", "watch_id": "bugs", "query": "bug OR error", "ingest_run_id": "run_def456", "hits": [{"chunk_id": "abc123", "score": 0.78, "snippet": "..."}]}
```

**Never read by the system** — purely for user review/audit.

### CLI Commands

#### `chinvex state generate --context <n> [--llm] [--since <duration>]`

1. Extract recently_changed from fingerprints
2. Extract active_threads from Codex docs
3. Extract TODOs from recently changed files
4. Run watch queries on new chunks
5. Write state.json
6. Render STATE.md

`--since 24h` or `--since 7d` limits "recently changed" window.

#### `chinvex state show --context <n>`

Print STATE.md to stdout.

#### `chinvex state open --context <n>`

Open STATE.md in default editor (QoL).

#### `chinvex state diff --context <n>`

Show what changed since last generation.

#### `chinvex state note add --context <n> "<text>"`

Add a human annotation to state.json.

#### `chinvex state note list --context <n>`

List all annotations.

#### `chinvex state note remove --context <n> --id <id>`

Remove an annotation.

### Post-Ingest Hook

#### IngestRunResult (Required)

Ingest must return this so post-ingest hook knows what's new:

```python
@dataclass
class IngestRunResult:
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

#### Hook Implementation

```python
def post_ingest_hook(context: Context, result: IngestRunResult):
    """
    Called after every ingest run.
    Receives IngestRunResult with new_chunk_ids for watch execution.

    State generation failures DO NOT fail ingest (best-effort).
    """
    try:
        # Generate new state
        state = generate_state(context, since_run=result.run_id)

        # Run watches ONLY on new chunks (not full corpus)
        if result.new_chunk_ids:
            watch_hits = run_watches(context, result.new_chunk_ids)
            state.watch_hits = watch_hits

            # Append to watch history log
            append_watch_history(context, result.run_id, watch_hits)

        state.generation_status = "ok"
        state.generation_error = None

    except Exception as e:
        log.error(f"State generation failed: {e}")
        # Keep old state or create minimal state
        state = load_previous_state(context) or create_empty_state(context)
        state.generation_status = "failed"
        state.generation_error = str(e)

    # Save (always, even on error)
    save_state_json(context, state)

    # Render
    md = render_state_md(state)
    save_state_md(context, md)

    if state.generation_status == "ok":
        log.info(f"STATE.md updated: {len(result.new_chunk_ids)} new chunks, {len(state.watch_hits)} watch hits")
    else:
        log.warning(f"STATE.md updated with errors: {state.generation_error}")


def append_watch_history(context: Context, run_id: str, watch_hits: list[WatchHit]):
    """Append watch hits to watch_history.jsonl."""
    history_path = context.dir / "watch_history.jsonl"
    with open(history_path, "a") as f:
        for hit in watch_hits:
            entry = {
                "timestamp": hit.triggered_at.isoformat(),
                "watch_id": hit.watch_id,
                "query": hit.query,
                "ingest_run_id": run_id,
                "hits": hit.hits
            }
            f.write(json.dumps(entry) + "\n")
```

### Acceptance Tests

```bash
# Test 1: State generation
chinvex state generate --context Chinvex
# Expected: state.json and STATE.md created

# Test 2: Recently changed appears
# (modify a file, run ingest, generate state)
chinvex state show --context Chinvex
# Expected: Modified file appears under "Recently Changed"

# Test 3: TODOs extracted
# (add "TODO: test" to a file, ingest)
chinvex state show --context Chinvex
# Expected: TODO appears under "TODOs"

# Test 4: Active threads shown
# (after Codex ingest)
chinvex state show --context Chinvex
# Expected: Recent Codex sessions appear under "Active Threads"
```

---

## 3. Recency Decay (P1.3)

### Purpose

Old stuff should fade. A decision from last week matters more than one from last year.

### Implementation

Apply recency decay as a **post-rank prior**, after source weights:

```python
from datetime import datetime, timezone

def recency_factor(updated_at: datetime, half_life_days: int = 90) -> float:
    """
    Exponential decay. Score halves every `half_life_days`.

    Returns value in (0, 1].

    IMPORTANT: Use UTC everywhere to avoid DST/timezone bugs.
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

**Timezone rule:** All timestamps stored as ISO8601 UTC. All comparisons in UTC.

### Chunk-Level Timestamps

Recency operates at **chunk granularity**, not document granularity:

- **For files (repo, notes):** Use document `updated_at` (file mtime)
- **For conversations:** Use **turn timestamp** stored in `chunks.metadata_json`:

```python
# When chunking conversations
chunk_metadata = {
    "turn_id": turn.turn_id,
    "turn_ts": turn.ts,  # ISO8601 timestamp of this specific turn
    ...
}
```

Retrieval fetches `updated_at` from chunk metadata:

```python
# Option A: Store in chunks table (P2)
# ALTER TABLE chunks ADD COLUMN updated_at TEXT;

# Option B: JOIN through documents (P1)
SELECT c.*, d.updated_at, json_extract(c.metadata_json, '$.turn_ts') as turn_ts
FROM chunks c
JOIN documents d ON c.doc_id = d.doc_id

# Use turn_ts if present, else fall back to d.updated_at
updated_at = turn_ts or d.updated_at
```

### Ranking Pipeline (Updated)

```python
def compute_final_rank(
    blended_score: float,
    source_type: str,
    updated_at: datetime,
    weights: dict,
    half_life_days: int = 90
) -> float:
    """
    Full ranking with source weight and recency decay.
    """
    source_weight = weights.get(source_type, 0.5)
    recency = recency_factor(updated_at, half_life_days)
    
    return blended_score * source_weight * recency
```

### Configuration

Add to `context.json`:

```json
{
  "ranking": {
    "recency_half_life_days": 90,
    "recency_enabled": true
  }
}
```

### CLI Flag

```bash
# Disable recency for historical research
chinvex search --context Chinvex "old decision" --no-recency
```

### Acceptance Tests

```bash
# Test 1: Recency affects ranking
# Create two docs with same content, different dates
# Query should rank newer one higher

# Test 2: --no-recency flag
chinvex search --context Chinvex "test" --no-recency
# Expected: Older docs not penalized

# Test 3: Very old content still findable
chinvex search --context Chinvex "ancient decision from 2024"
# Expected: Results appear (just ranked lower)
```

---

## 4. Watch List (P1.4)

### Purpose

"Tell me when something changes about X." Proactive alerts without building notification infrastructure.

### watch.json Schema

Location: `P:\ai_memory\contexts\<Context>\watch.json`

```json
{
  "schema_version": 1,
  "watches": [
    {
      "id": "p2_https",
      "query": "P2 HTTPS endpoint",
      "min_score": 0.75,
      "enabled": true,
      "created_at": "2026-01-26T12:00:00Z"
    },
    {
      "id": "bugs",
      "query": "bug OR error OR traceback",
      "min_score": 0.75,
      "enabled": true,
      "created_at": "2026-01-26T12:00:00Z"
    }
  ]
}
```

**Default min_score:** 0.75 (raise if too noisy, lower if too quiet).

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier for the watch |
| `query` | string | yes | Search query to run |
| `min_score` | float | yes | Minimum blended score to trigger |
| `enabled` | bool | yes | Whether watch is active |
| `created_at` | ISO8601 | yes | When watch was created |

### Execution

Watches run **only on newly ingested chunks**, not the full corpus:

```python
def run_watches(
    context: Context,
    new_chunk_ids: list[str],
    timeout_per_watch: int = 30
) -> list[WatchHit]:
    """
    Run all enabled watches against newly ingested chunks.

    Timeout per watch: 30s (skip watch if exceeded).
    """
    watches = load_watches(context)
    hits = []

    for watch in watches:
        if not watch.enabled:
            continue

        try:
            with timeout(timeout_per_watch):
                # Search only new chunks
                results = search_chunks(
                    context=context,
                    query=watch.query,
                    chunk_ids=new_chunk_ids,  # Filter to new only
                    k=10
                )

                # Filter by min_score
                matching = [r for r in results if r.blended_score >= watch.min_score]

                if matching:
                    hits.append(WatchHit(
                        watch_id=watch.id,
                        query=watch.query,
                        hits=[
                            {"chunk_id": r.chunk_id, "score": r.blended_score, "snippet": r.text[:200]}
                            for r in matching[:5]
                        ],
                        triggered_at=datetime.now()
                    ))

        except TimeoutError:
            log.warning(f"Watch {watch.id} timed out after {timeout_per_watch}s, skipping")
            continue  # Skip this watch, don't fail entire run
        except Exception as e:
            log.error(f"Watch {watch.id} failed: {e}")
            continue  # Best-effort

    return hits
```

### Integration with STATE.md

Watch hits are written to `state.json` and rendered in STATE.md:

```markdown
## Watch Hits (since last ingest)

### p2_https
- [0.85] ...discussion of HTTPS endpoint for ChatGPT...
- [0.78] ...API authentication design for external access...

### bugs
- [0.82] ...traceback in Chroma connection...
```

### CLI Commands

#### `chinvex watch add --context <n> --id <id> --query <q> [--min-score 0.75]`

Add a new watch. Validates:
- ID is unique (error if collision, don't overwrite)
- ID contains only alphanumeric, underscore, hyphen
- Query is non-empty

#### `chinvex watch list --context <n>`

List all watches with status (id, query, min_score, enabled).

#### `chinvex watch remove --context <n> --id <id>`

Remove a watch by ID.

#### `chinvex watch run --context <n> [--all]`

Manually run watches. `--all` runs against full corpus (not just new chunks).

#### `chinvex watch dry-run --context <n> --query <q> [--min-score 0.75] [--top 20]`

Preview what a watch query would match against current index (top N results with scores).

Does NOT require watch to be added first. Use for calibration.

### Acceptance Tests

```bash
# Test 1: Add watch
chinvex watch add --context Chinvex --id test_watch --query "retrieval"
# Expected: Watch added to watch.json

# Test 2: Watch triggers on ingest
# (add a doc containing "retrieval", run ingest)
chinvex state show --context Chinvex
# Expected: Watch hit appears under "Watch Hits"

# Test 3: Watch doesn't trigger below threshold
# (add watch with min_score=0.99)
# Expected: No hits unless very strong match

# Test 4: List watches
chinvex watch list --context Chinvex
# Expected: Table of watches with id, query, min_score, enabled
```

---

## 5. STATE.md LLM Consolidator (P1.5)

### Purpose

Optional enhancement that uses an LLM to:
- Deduplicate similar items
- Extract decisions/facts from content
- Detect conflicts ("X was decided, but later Y contradicts it")

### Trigger

```bash
# Explicit flag required
chinvex state generate --context Chinvex --llm
```

### What the LLM Does

Given `state.json` + recent chunk content, produce:

```json
{
  "decisions": [
    {
      "text": "Hybrid retrieval uses FTS5 + Chroma",
      "source_chunk_ids": ["abc123"],
      "confidence": 0.9,
      "date": "2026-01-20"
    }
  ],
  "facts": [
    {
      "text": "Ollama runs on skynet:11434",
      "source_chunk_ids": ["def456"],
      "confidence": 0.95
    }
  ],
  "conflicts": [
    {
      "a": "Use source weights in blend",
      "b": "Apply source weights post-retrieval",
      "resolution": "B supersedes A (2026-01-26)",
      "source_chunk_ids": ["abc", "def"]
    }
  ]
}
```

### Prompt Template

**Context budget:** Last 20 chunks from recently_changed, truncated to 500 chars each.
**Total prompt budget:** ~4k tokens (fits in any model context window).

```
You are analyzing recent activity in a personal knowledge base.

Given the following recently changed content (last 20 chunks):
{chunk_summaries}

And the current state:
{current_state_json}

Extract:
1. DECISIONS: Explicit choices that were made (e.g., "we decided to use X")
2. FACTS: Factual information worth remembering (e.g., "server runs on port 8080")
3. CONFLICTS: Cases where newer content contradicts older content

For each, cite the source chunk_id and provide a confidence score (0.0-1.0).

Respond in JSON format:
{schema}
```

**Confidence thresholds:**
- Confidence < 0.7: Render with "(low confidence)" in STATE.md
- Conflicts: Best-effort, no guarantees (high false positive risk acceptable)

### Configuration

```json
{
  "state_llm": {
    "enabled": false,
    "model": "llama3:8b",
    "ollama_url": "http://skynet:11434"
  }
}
```

### Acceptance Tests

```bash
# Test 1: LLM consolidation produces decisions
chinvex state generate --context Chinvex --llm
# Expected: state.json has "decisions" populated

# Test 2: Conflicts detected
# (ingest contradictory content)
chinvex state generate --context Chinvex --llm
# Expected: "conflicts" section populated

# Test 3: Graceful degradation
# (Ollama offline)
chinvex state generate --context Chinvex --llm
# Expected: Warning, falls back to deterministic-only
```

---

## 6. CLI Reference (P1 Additions)

### New Commands

| Command | Description |
|---------|-------------|
| `chinvex codex health` | Check app-server connectivity |
| `chinvex codex list` | List Codex threads |
| `chinvex state generate` | Generate state.json + STATE.md |
| `chinvex state show` | Print STATE.md |
| `chinvex state open` | Open STATE.md in editor |
| `chinvex state diff` | Show changes since last generation |
| `chinvex state note add` | Add human annotation |
| `chinvex state note list` | List annotations |
| `chinvex state note remove` | Remove annotation |
| `chinvex watch add` | Add a watch |
| `chinvex watch list` | List watches |
| `chinvex watch remove` | Remove a watch |
| `chinvex watch run` | Manually run watches |
| `chinvex watch dry-run` | Preview watch query results for calibration |

### Updated Commands

| Command | Change |
|---------|--------|
| `chinvex ingest` | Now calls post-ingest hook (state generation) |
| `chinvex search` | Now applies recency decay (add `--no-recency` flag) |

---

## 7. Implementation Order (Revised)

### Phase 0: Foundation (Unblocks Everything)
**Task 1: Add IngestRunResult** (30 min)
- Refactor `ingest.py` to return `IngestRunResult` dataclass
- Track `new_doc_ids`, `updated_doc_ids`, `new_chunk_ids`, `skipped_doc_ids`, `error_doc_ids`
- This unblocks P1.2 (post-ingest hook needs it)

### Phase 1: Parallel Development (After Task 1)

**P1.1: Real Codex Ingestion** (can proceed independently)
1. Port client code from godex (auth included)
2. Capture real API samples
3. Finalize schemas from samples
4. Implement pagination and error handling
5. Wire up to ingest pipeline
6. Test end-to-end

**P1.2: STATE.md Deterministic** (can proceed in parallel)
1. Upgrade context.json schema to v2 (auto-migration)
2. Define state.json schema with generation_status/error
3. Implement extractors (recently_changed, todos, active_threads with empty handling)
4. Implement renderer with "do not edit" header
5. Add post-ingest hook with failure handling
6. Add watch_history.jsonl append logic
7. CLI commands

**P1.3: Recency Decay** (small, can proceed in parallel)
1. Add ranking config to context.json v2
2. Implement recency_factor with naive timestamp handling
3. Integrate into ranking pipeline (chunk-level timestamps)
4. Add --no-recency CLI flag
5. Test ranking changes

### Phase 2: Watch List (Depends on P1.2)
**P1.4: Watch List** (needs state generation)
1. Define watch.json schema
2. Implement watch runner with timeout and error handling
3. Add watch ID uniqueness validation
4. Integrate with state generation
5. CLI commands (add, list, remove, run, dry-run)
6. Test watch triggering

### Phase 3: Optional Enhancement
**P1.5: LLM Consolidator** (optional, last)
1. Define LLM output schema
2. Implement prompt + parsing (last 20 chunks, 4k token budget)
3. Integrate with state generation (--llm flag)
4. Handle graceful degradation
5. Test decision/fact extraction

### Dependency Graph

```
P1.0: IngestRunResult
    ↓
┌───────────────────────────────────────┐
│  P1.1: Codex ingestion (parallel)     │
│  P1.2: STATE.md (parallel)            │
│  P1.3: Recency decay (parallel)       │
└───────────────────────────────────────┘
    ↓
P1.4: Watch list (needs P1.2)
    ↓
P1.5: LLM consolidator (optional)
```

**Key insight:** P1.2 can show "(Codex ingestion not configured)" for active_threads until P1.1 lands.

---

## 8. Acceptance Test Summary

| ID | Test | Pass Criteria |
|----|------|---------------|
| P1.1.1 | App-server health | Returns reachable status |
| P1.1.2 | List threads | Shows real thread summaries |
| P1.1.3 | Ingest threads | Threads appear in search |
| P1.1.4 | Fingerprint skip | Re-ingest skips unchanged |
| P1.2.1 | State generation | state.json + STATE.md created |
| P1.2.2 | Recently changed | Modified files appear |
| P1.2.3 | TODOs extracted | TODO comments appear |
| P1.2.4 | Active threads | Codex sessions appear |
| P1.3.1 | Recency ranking | Newer docs rank higher |
| P1.3.2 | No-recency flag | Old docs not penalized |
| P1.4.1 | Add watch | watch.json updated |
| P1.4.2 | Watch triggers | Hits appear in STATE.md |
| P1.4.3 | Min score filter | Low scores don't trigger |
| P1.5.1 | LLM decisions | Decisions extracted |
| P1.5.2 | LLM conflicts | Conflicts detected |
| P1.5.3 | LLM fallback | Works when Ollama offline |

---

## 9. Key Design Decisions

### Authentication
Port from godex verbatim. Auth handled inside godex client (cookies/token/session). P1 ships with "inherit from godex" only. Token config is P2.

### Context Schema Evolution
Bump to schema_version: 2 with auto-upgrade on load. Missing fields get defaults. No explicit migrate command.

### Error Handling Philosophy
Graceful degradation everywhere:
- State gen fails → ingest succeeds (best-effort)
- Watch execution fails → ingest succeeds (skip watch)
- Single thread fails → skip, continue to next

Fail-fast only for: DB connection, disk write, context doesn't exist.

### Thread Fingerprinting
Use `updated_at` only (not AND with `last_turn_id`). If app-server reports change but content identical, harmless re-ingest is acceptable. Content SHA hash is P2 "paranoia mode."

### Recency Decay Pipeline
`Blend → Source Weight → Recency → Final Rank`

This is correct. Relevance first, then priors. Order doesn't matter when multiplying, but conceptually clean.

### Recency Granularity
Chunk-level, not document-level. For conversations, use turn timestamp from metadata. For files, use document mtime.

### TODO Extraction
Accept false positives (TODOs in strings, not just comments). Skip files > 1MB. Use word boundaries in regex to reduce noise.

### Watch Hit Persistence
- `state.json.watch_hits`: Ephemeral (overwritten each ingest)
- `watch_history.jsonl`: Append-only audit log with timestamp + hits
- Prevents state.json growth, provides historical record

### Watch Timeouts
30s per watch. If timeout exceeded, log warning and skip watch (don't fail entire run).

### Watch ID Validation
User-provided IDs (not auto-generated UUIDs). Validate uniqueness on add. Error if collision (don't overwrite). Human-readable IDs preferred.

### LLM Consolidator Scope
P1.5 is optional. Pass last 20 chunks (truncated 500 chars each), 4k token budget. Confidence < 0.7 rendered as "(low confidence)." Conflict detection is best-effort.

### Parallel Implementation
P1.1, P1.2, P1.3 can proceed in parallel after P1.0 (IngestRunResult) lands. P1.4 depends on P1.2. P1.5 is last and optional.

---

*End of P1 spec.*
