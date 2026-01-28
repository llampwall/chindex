# Chinvex P3 Implementation Spec

**Version:** 1.3  
**Date:** 2026-01-28  
**Status:** Draft — for future reference  
**Depends on:** P2 complete

---

## Overview

P3 is the "quality of life" release. P2 got ChatGPT connected. P3 makes everything work better: smarter chunking, proactive alerts, search across all your stuff, and fading old content gracefully.

### Completed Early (in P2)

The following P3 items were shipped as part of P2 hardening:

- ✅ **Request ID tracking** — All responses include `X-Request-ID` header for correlation
- ✅ **Deep health check** — `GET /healthz` checks SQLite, Chroma, and context registry readiness (returns 503 if degraded)
- ✅ **Error logging** — Global exception handler logs stack traces to `P:\ai_memory\gateway_errors.jsonl` with request_id
- ✅ **Startup warmup** — Gateway preloads context registry and initializes SQLite/Chroma on startup (prevents cold-start 500s)

### P3 Phased Approach

Split into sub-releases for incremental delivery:

**P3a — Quality + Convenience** (highest leverage)
1. **P3.1** Chunking v2 (overlap, semantic boundaries, Python code-aware)
2. **P3.3** Cross-context search

**P3b — Proactive Foundation** (no network complexity)
3. **P3.2** Watch history log + CLI (webhooks deferred)

**P3c — Policy + Ops** (lower priority)
4. **P3.4** Archive tier
5. **P3.2b** Webhook notifications
6. **P3.5** Gateway extras (Redis rate limiting, Prometheus metrics)

### Non-Goals (P4+)

- Multi-user auth / OAuth
- PDF/email/Slack ingestion adapters
- Web UI dashboard
- Mobile app
- Real-time sync / live updates
- Conversation memory (multi-turn context in gateway)

---

## 1. Chunking Strategy Improvements (P3.1)

### Current State (P0-P2)

- Fixed character limit (~3000 chars)
- Repo files: simple overlap (300 chars)
- Conversations: never split turns
- No semantic awareness

### Problems

1. Hard splits break context — function split mid-body
2. No preference for natural boundaries
3. Code-unaware — doesn't respect function/class structure

### P3.1 Scope

**In scope:**
- Overlap for all sources
- Semantic boundary detection (markdown headers, blank lines)
- Python code-aware splitting (AST-based)
- JS/TS heuristic splitting (regex for function/class/export — NOT full parsing)

**Out of scope (future):**
- True JS/TS code-aware splitting with tree-sitter/babel
- Other languages

### Improved Strategy

#### 1.1 Overlap (All Sources)

Always include overlap to recover context at chunk boundaries:

```python
CHUNK_SIZE = 3000      # chars
OVERLAP = 300          # chars (~10%)
```

**Implementation:**
```python
def chunk_with_overlap(text: str, size: int = 3000, overlap: int = 300) -> list[tuple[int, int]]:
    """Return list of (start, end) positions for chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return chunks
```

#### 1.2 Semantic Boundary Detection

Try to split at natural boundaries, falling back gracefully:

```python
SPLIT_PRIORITIES = [
    (r'\n## ', 100),           # Markdown H2
    (r'\n### ', 90),           # Markdown H3
    (r'\n---\n', 85),          # Markdown horizontal rule
    (r'\nclass ', 80),         # Python class
    (r'\ndef ', 75),           # Python function
    (r'\nasync def ', 75),     # Python async function
    (r'\nfunction ', 75),      # JS function (heuristic)
    (r'\nconst \w+ = ', 70),   # JS const declaration (heuristic)
    (r'\nexport ', 70),        # JS/TS export (heuristic)
    (r'\n\n\n', 60),           # Multiple blank lines
    (r'\n\n', 50),             # Paragraph break
    (r'\n', 10),               # Line break (last resort)
]

def find_best_split(text: str, target_pos: int, size: int = 3000) -> int:
    """
    Find best split point near target_pos.
    Searches within ±window chars for highest-priority boundary.
    Window is proportional to chunk size to handle large code blocks.
    Returns position of the boundary (start of the pattern match).
    """
    window = min(800, max(300, size // 10))  # Proportional window
    search_start = max(0, target_pos - window)
    search_end = min(len(text), target_pos + window)
    search_region = text[search_start:search_end]
    
    best_pos = target_pos
    best_priority = 0
    
    for pattern, priority in SPLIT_PRIORITIES:
        for match in re.finditer(pattern, search_region):
            match_pos = search_start + match.start()
            # Prefer boundaries closer to target
            distance_penalty = abs(match_pos - target_pos) / window * 10
            effective_priority = priority - distance_penalty
            if effective_priority > best_priority:
                best_priority = effective_priority
                best_pos = match_pos
    
    return best_pos
```

#### 1.3 Code-Aware Splitting

For code files, extract logical boundaries first:

```python
import ast

def extract_python_boundaries(text: str) -> list[int]:
    """Extract line numbers where top-level definitions start."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    
    boundaries = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            boundaries.append(node.lineno - 1)  # 0-indexed
    return boundaries

def line_to_char_pos(text: str, line_num: int) -> int:
    """Convert line number to character position."""
    lines = text.split('\n')
    return sum(len(line) + 1 for line in lines[:line_num])
```

**Chunking algorithm (clarified):**

Boundaries define where chunks *can* start. We accumulate text until near `max_chars`, then flush at the most recent boundary.

```python
def chunk_python_file(text: str, max_chars: int = 3000) -> list[Chunk]:
    """
    Split Python file respecting function/class boundaries.
    
    Algorithm:
    1. Get boundary positions (where functions/classes start)
    2. Walk through boundaries, accumulating segments
    3. When accumulated size exceeds max_chars, flush previous chunk
    4. Each chunk starts at a boundary position
    """
    boundaries = extract_python_boundaries(text)
    boundary_positions = [0] + [line_to_char_pos(text, ln) for ln in boundaries] + [len(text)]
    
    if len(boundary_positions) <= 2:
        # No internal boundaries, fallback to generic
        return chunk_with_overlap_and_boundaries(text)
    
    chunks = []
    chunk_start = 0
    
    for i, pos in enumerate(boundary_positions[1:], 1):
        segment_size = pos - chunk_start
        
        if segment_size > max_chars and chunk_start != boundary_positions[i-1]:
            # Flush chunk ending at previous boundary
            chunk_end = boundary_positions[i-1]
            chunks.append(make_chunk(text[chunk_start:chunk_end], chunk_start))
            chunk_start = chunk_end
    
    # Final chunk
    if chunk_start < len(text):
        chunks.append(make_chunk(text[chunk_start:], chunk_start))
    
    return chunks
```

#### 1.4 Language Detection

```python
LANGUAGE_MAP = {
    '.py': 'python',
    '.pyw': 'python',
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.jsx': 'javascript',
    '.md': 'markdown',
    '.mdx': 'markdown',
}

def detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return LANGUAGE_MAP.get(ext, 'generic')

def chunk_file(text: str, filepath: str) -> list[Chunk]:
    language = detect_language(filepath)
    
    if language == 'python':
        return chunk_python_file(text)
    elif language in ('javascript', 'typescript'):
        return chunk_js_file(text)
    elif language == 'markdown':
        return chunk_markdown_file(text)
    else:
        return chunk_generic_file(text)
```

### Configuration

```json
{
  "chunking": {
    "max_chars": 3000,
    "max_lines": 300,  // For code: whichever limit hits first
    "overlap_chars": 300,
    "semantic_boundaries": true,
    "code_aware": true,
    "skip_files_larger_than_mb": 5
  }
}
```

**Note on units:** For prose (markdown, docs), `max_chars` is the primary limit. For code, use whichever of `max_chars` or `max_lines` is reached first — this handles both minified code (char-heavy) and verbose code (line-heavy).

### Migration

Bump `chunker_version` in fingerprints. On next ingest, files will be re-chunked automatically.

```python
CHUNKER_VERSION = 2  # Was 1 in P0-P2
```

#### Rechunk-Only Mode

To avoid re-embedding unchanged text, add `--rechunk-only` flag:

```bash
# Full reingest (rechunk + re-embed)
chinvex ingest --context Chinvex --full

# Rechunk only (reuse embeddings if text hash unchanged)
chinvex ingest --context Chinvex --rechunk-only
```

**Implementation:**

Use a stable per-chunk key for embedding reuse:

```python
import hashlib

def chunk_key(text: str) -> str:
    """Stable key for embedding lookup. Normalize whitespace first."""
    normalized = ' '.join(text.split())  # Collapse whitespace
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
```

**Rechunk flow:**
1. Run new chunker on source file
2. For each new chunk, compute `chunk_key(text)`
3. Check Chroma for existing embedding with that key
4. If found and text hash matches, reuse embedding
5. If not found or hash mismatch, generate new embedding
6. Log stats: `Rechunked: 142, Reused embeddings: 98, New embeddings: 44`

This prevents "why is my machine melting?" on large contexts.

### Acceptance Tests

```bash
# Test 3.1.1: Function not split mid-body
# Ingest Python file with 50-line function
# Verify function is in single chunk OR split at nested function/class

# Test 3.1.2: Markdown splits at headers
# Ingest markdown with ## sections
# Verify chunks start at or near ## boundaries

# Test 3.1.3: Overlap present
# Get two consecutive chunks
# Verify last ~300 chars of chunk N appear in chunk N+1

# Test 3.1.4: Large files handled
# Ingest 10MB file
# Verify it's either chunked or skipped (not crashed)
```

---

## 2. Watch History + Webhook Notifications (P3.2)

### Current State (P1)

- Watch hits stored in `state.json` (ephemeral, overwritten each ingest)
- No persistent history
- No notification mechanism

### P3 Additions

#### 2.1 Watch History Log

Append-only log of all watch triggers:

**File:** `P:\ai_memory\contexts\<Context>\watch_history.jsonl`

```jsonl
{"ts": "2026-01-27T04:00:00Z", "run_id": "run_abc", "watch_id": "p2_https", "query": "P2 HTTPS", "hits": [{"chunk_id": "abc123", "score": 0.85, "snippet": "...first 200 chars..."}]}
{"ts": "2026-01-27T05:00:00Z", "run_id": "run_def", "watch_id": "bugs", "query": "bug error", "hits": [{"chunk_id": "def456", "score": 0.78, "snippet": "..."}]}
```

**Fields:**
- `ts`: ISO8601 timestamp
- `run_id`: Ingest run that triggered the watch
- `watch_id`: Watch identifier
- `query`: Watch query
- `hits`: Array of matching chunks (snippet is first 200 chars)

#### 2.2 CLI Commands

```bash
# View watch history
chinvex watch history --context Chinvex [--since 7d] [--id p2_https] [--limit 50]

# Output formats
chinvex watch history --context Chinvex --format json
chinvex watch history --context Chinvex --format table  # default

# Clear old history
chinvex watch history clear --context Chinvex --older-than 90d
```

#### 2.3 Webhook Notifications

Fire HTTP POST when watch triggers.

**Configuration:**
```json
{
  "notifications": {
    "enabled": false,  // OFF by default for safety
    "webhook_url": "https://your-webhook.example.com/chinvex",
    "webhook_secret": "env:CHINVEX_WEBHOOK_SECRET",
    "notify_on": ["watch_hit"],
    "min_score_for_notify": 0.75,
    "retry_count": 2,
    "retry_delay_sec": 5
  }
}
```

**Payload:**
```json
{
  "event": "watch_hit",
  "timestamp": "2026-01-27T04:00:00Z",
  "context": "Chinvex",
  "run_id": "run_abc123",
  "watch": {
    "id": "p2_https",
    "query": "P2 HTTPS endpoint"
  },
  "hits": [
    {
      "chunk_id": "abc123",
      "score": 0.85,
      "source_uri": "...",
      "snippet": "...first 200 chars..."
    }
  ],
  "signature": "sha256=..."
}
```

**Security note:** Webhook payload includes snippet (first 200 chars) and source_uri, but NEVER full chunk text. This prevents accidental data leakage via webhook logs.
```

**Signature verification:**
```python
import hmac
import hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = 'sha256=' + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

#### 2.4 Notification Destinations (Future)

P3 ships with webhook only. Future options:
- Discord webhook (just a different URL format)
- Slack incoming webhook
- Email (requires SMTP config)
- Desktop notification (requires native integration)

### Acceptance Tests

```bash
# Test 3.2.1: History appends
chinvex ingest --context Chinvex
chinvex ingest --context Chinvex  # second run
chinvex watch history --context Chinvex
# Expected: Entries from both runs

# Test 3.2.2: History filtering
chinvex watch history --context Chinvex --since 1h --id p2_https
# Expected: Only recent, matching entries

# Test 3.2.3: Webhook fires
# Configure webhook to requestbin.com or similar
# Trigger watch, verify POST received with correct payload

# Test 3.2.4: Webhook signature valid
# Verify signature in received payload matches expected
```

---

## 3. Cross-Context Search (P3.3)

### Purpose

Search across multiple contexts at once. Useful when you don't remember which project something was in.

### CLI Interface

```bash
# Search all contexts
chinvex search --all "that authentication bug"

# Search specific contexts
chinvex search --contexts Chinvex,Personal "OAuth flow"

# Exclude contexts
chinvex search --all --exclude Work "personal project"
```

### API Endpoint

Extend `/v1/search` and `/v1/evidence`:

```json
// POST /v1/search
{
  "contexts": ["Chinvex", "Personal"],  // or "all"
  "query": "authentication",
  "k": 10  // total results, not per-context
}
```

**Response:**
```json
{
  "query": "authentication",
  "contexts_searched": ["Chinvex", "Personal"],
  "results": [
    {
      "context": "Chinvex",
      "chunk_id": "abc123",
      "text": "...",
      "score": 0.89,
      ...
    },
    {
      "context": "Personal",
      "chunk_id": "def456",
      "text": "...",
      "score": 0.85,
      ...
    }
  ],
  "total_results": 10
}
```

### Implementation

```python
def search_multi_context(
    contexts: list[str] | Literal["all"],
    query: str,
    k: int = 10
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.
    """
    if contexts == "all":
        contexts = list_all_contexts()
    
    # Cap contexts to prevent slowdown
    contexts = contexts[:config.cross_context.max_contexts_per_query]
    
    # Per-context cap: don't fetch more than needed
    k_per_context = min(k, 20)
    
    # Gather results from each context
    all_results = []
    for ctx_name in contexts:
        ctx = load_context(ctx_name)
        results = search_context(ctx, query, k=k_per_context)
        for r in results:
            r.context = ctx_name  # Tag with source context
        all_results.extend(results)
    
    # Sort by score descending, take top k
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:k]
```

### Configuration

```json
{
  "cross_context": {
    "enabled": true,
    "max_contexts_per_query": 10,
    "k_per_context": 20,
    "default_contexts": ["Chinvex", "Personal"]  // for "all" shorthand
  }
}
```

### Gateway Security

Cross-context search respects allowlist:
- If `context_allowlist` is set, `"all"` means "all allowed contexts"
- Cannot search contexts not in allowlist

### Score Comparability

**Note:** Merging by raw `score` assumes scores are comparable across contexts. Chinvex normalizes scores to 0–1 per-query per-index, so this is *mostly* safe. However, if you observe ranking anomalies:

```python
# Optional: re-normalize after merge if scores aren't comparable
def normalize_merged_scores(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return results
    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    range_score = max_score - min_score or 1.0
    for r in results:
        r.score = (r.score - min_score) / range_score
    return results
```

For P3, assume scores are comparable. Add re-normalization only if testing reveals issues.

### Acceptance Tests

```bash
# Test 3.3.1: --all searches all contexts
chinvex search --all "test"
# Expected: Results from multiple contexts

# Test 3.3.2: Results tagged with context
# Verify each result shows context name

# Test 3.3.3: API multi-context
curl -X POST /v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"contexts": ["Chinvex", "Personal"], "query": "test"}'
# Expected: Mixed results with context tags

# Test 3.3.4: Allowlist respected
# Set allowlist to ["Chinvex"]
# Request contexts: ["Chinvex", "Personal"]
# Expected: Only Chinvex results (Personal silently excluded or error)
```

---

## 4. Archive Tier (P3.4)

### Purpose

Old content clutters search results. Archive tier moves stale content out of default search while keeping it accessible on request.

### Mechanics

```
┌─────────────────┐
│  Active Index   │  ← Default search
│  (recent docs)  │
└────────┬────────┘
         │ age > threshold
         ▼
┌─────────────────┐
│  Archived       │  ← Search with --include-archive
│  (old docs)     │
└─────────────────┘
```

### Archive vs Recency Decay Interaction

**Precedence rules:**
1. Default search excludes archived docs entirely (hard filter)
2. Recency decay applies only within active (non-archived) docs
3. `--include-archive` reintroduces archived docs with a score penalty (e.g., 0.8x multiplier)

This means:
- Very old docs (>180d) get archived → completely out of default results
- Moderately old active docs (30-180d) → still searchable, but recency-decayed
- `--include-archive` brings back archived docs, but they rank lower than equally-relevant active docs

### Implementation: Flag-Based

Add `archived` column to documents table (simpler than separate indexes):

```sql
ALTER TABLE documents ADD COLUMN archived INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN archived_at TEXT;  -- ISO8601
CREATE INDEX idx_documents_archived ON documents(archived);
```

**Search modification:**
```python
def search_context(ctx, query, k=10, include_archive=False):
    # ... existing search logic ...
    
    if not include_archive:
        # Filter out archived docs at the SQL level
        where_clause += " AND d.archived = 0"
    else:
        # Apply archive penalty to scores
        for result in results:
            if result.archived:
                result.score *= 0.8  # Archive penalty
    
    # ... rest of search ...
```
```

### Configuration

```json
{
  "archive": {
    "enabled": true,
    "age_threshold_days": 180,
    "auto_archive_on_ingest": true
  }
}
```

### Archive Timestamp Source

**Which clock determines "age"?**

- **Repo files:** Use `updated_at` (file mtime from source)
- **Chat threads:** Use `updated_at` (last message timestamp)
- **Codex sessions:** Use `updated_at` (session end time)
- **Fallback:** If `updated_at` is missing, use `ingested_at`

```python
def get_doc_age_timestamp(doc: Document) -> datetime:
    """Get the timestamp used for archive age calculation."""
    return doc.updated_at or doc.ingested_at
```

This ensures archive reflects *content* age, not just when you happened to ingest it.

### CLI Commands

```bash
# Manual archive (mark docs older than threshold)
chinvex archive run --context Chinvex [--older-than 180d] [--dry-run]

# Search including archived
chinvex search --context Chinvex "old decision" --include-archive

# List archived documents
chinvex archive list --context Chinvex [--limit 50]

# Restore specific document
chinvex archive restore --context Chinvex --doc-id abc123

# Restore by query (interactive)
chinvex archive restore --context Chinvex --query "project X" --interactive

# Permanently delete archived (careful!)
chinvex archive purge --context Chinvex --older-than 365d [--confirm]
```

### API Extension

```json
// POST /v1/search
{
  "context": "Chinvex",
  "query": "old project",
  "include_archive": true
}

// POST /v1/evidence
{
  "context": "Chinvex",
  "query": "that decision from last year",
  "include_archive": true
}
```

### Auto-Archive on Ingest

If `auto_archive_on_ingest` is true, archive check runs after each ingest:

```python
def post_ingest_hook(ctx, result):
    # ... existing state generation ...
    
    if ctx.config.archive.enabled and ctx.config.archive.auto_archive_on_ingest:
        archive_old_documents(ctx, ctx.config.archive.age_threshold_days)
```

### Acceptance Tests

```bash
# Test 3.4.1: Auto-archive runs
# Ingest doc with old updated_at (200 days ago)
chinvex ingest --context Chinvex
chinvex archive list --context Chinvex
# Expected: Old doc in archived list

# Test 3.4.2: Archived excluded by default
chinvex search --context Chinvex "old content"
# Expected: Archived doc not in results

# Test 3.4.3: --include-archive finds it
chinvex search --context Chinvex "old content" --include-archive
# Expected: Archived doc in results (marked as archived)

# Test 3.4.4: Restore works
chinvex archive restore --context Chinvex --doc-id abc123
chinvex search --context Chinvex "old content"
# Expected: Doc now in normal results

# Test 3.4.5: Purge removes permanently
chinvex archive purge --context Chinvex --older-than 365d --confirm
# Expected: Very old docs gone from both active and archive
```

---

## 5. Gateway Improvements (P3.5)

### 5.1 Redis-Backed Rate Limiting

P2 uses in-memory rate limiting (resets on restart). P3 adds Redis for persistence.

```json
{
  "gateway": {
    "rate_limit": {
      "backend": "redis",  // or "memory"
      "redis_url": "redis://localhost:6379/0",
      "requests_per_minute": 60,
      "requests_per_hour": 500
    }
  }
}
```

**Fallback:** If Redis unavailable, fall back to in-memory with warning.

### 5.2 Prometheus Metrics Endpoint

```
GET /metrics
```

Returns Prometheus-format metrics:

```
# HELP chinvex_requests_total Total requests by endpoint and status
# TYPE chinvex_requests_total counter
chinvex_requests_total{endpoint="/v1/evidence",status="200"} 1542
chinvex_requests_total{endpoint="/v1/evidence",status="401"} 23
chinvex_requests_total{endpoint="/v1/search",status="200"} 892

# HELP chinvex_request_duration_seconds Request latency
# TYPE chinvex_request_duration_seconds histogram
chinvex_request_duration_seconds_bucket{endpoint="/v1/evidence",le="0.1"} 1200
chinvex_request_duration_seconds_bucket{endpoint="/v1/evidence",le="0.5"} 1500
chinvex_request_duration_seconds_bucket{endpoint="/v1/evidence",le="1.0"} 1540

# HELP chinvex_grounded_ratio Ratio of grounded=true responses
# TYPE chinvex_grounded_ratio gauge
chinvex_grounded_ratio 0.73
```

**Configuration:**
```json
{
  "gateway": {
    "metrics_enabled": true,
    "metrics_auth_required": false  // Usually internal only
  }
}
```

### ~~5.3 Request ID Tracking~~ ✅ Completed in P2

### ~~5.4 Deep Health Check~~ ✅ Completed in P2

### Acceptance Tests

```bash
# Test 3.5.1: Redis rate limiting persists
# Hit rate limit, restart gateway, verify still limited

# Test 3.5.2: Metrics endpoint
curl http://localhost:7778/metrics
# Expected: Prometheus-format metrics
```

---

## 6. Implementation Order

Matches P3a/P3b/P3c phasing from overview.

### P3a: Quality + Convenience

#### Phase 1: Chunking (P3.1) — 2-3 days
1. Overlap implementation
2. Semantic boundary detection (proportional window)
3. Python code-aware splitting (AST-based)
4. JS/TS heuristic splitting (regex only)
5. Markdown-aware splitting
6. Language detection
7. Rechunk-only mode with stable chunk keys
8. Bump chunker_version
9. Re-index test + stats logging

#### Phase 2: Cross-Context Search (P3.3) — 1-2 days
10. Multi-context search logic
11. Result merging by score (with per-context caps)
12. CLI --all / --contexts flags
13. API multi-context support
14. Allowlist enforcement

### P3b: Proactive Foundation

#### Phase 3: Watch History (P3.2 — log only) — 1 day
15. watch_history.jsonl append on trigger
16. `chinvex watch history` CLI
17. History filtering (--since, --id)

### P3c: Policy + Ops

#### Phase 4: Archive Tier (P3.4) — 2 days
18. Add archived column to schema
19. Archive timestamp source logic (updated_at vs ingested_at)
20. Archive run command
21. Search filtering + archive penalty
22. Archive list command
23. Restore command
24. Auto-archive on ingest
25. Purge command

#### Phase 5: Webhooks (P3.2b) — 1 day
26. Webhook notification implementation
27. Webhook signature generation
28. Retry logic
29. Security hardening (off by default, snippet only)

#### Phase 6: Gateway Extras (P3.5) — 1-2 days
30. Redis rate limiting
31. Prometheus metrics endpoint

---

## 7. Dependencies

### New Packages

```
redis>=5.0.0  # For rate limiting (optional)
prometheus-client>=0.19.0  # For metrics (optional)
```

### Existing (No Changes)

- fastapi
- uvicorn
- chromadb
- sqlite3

---

## 8. Configuration Reference

### Complete P3 Config

```json
{
  "chunking": {
    "max_chars": 3000,
    "overlap_chars": 300,
    "semantic_boundaries": true,
    "code_aware": true,
    "skip_files_larger_than_mb": 5
  },
  "notifications": {
    "enabled": false,
    "webhook_url": "",
    "webhook_secret": "env:CHINVEX_WEBHOOK_SECRET",
    "notify_on": ["watch_hit"],
    "min_score_for_notify": 0.75,
    "retry_count": 2,
    "retry_delay_sec": 5
  },
  "cross_context": {
    "enabled": true,
    "max_contexts_per_query": 10
  },
  "archive": {
    "enabled": true,
    "age_threshold_days": 180,
    "auto_archive_on_ingest": true
  },
  "gateway": {
    "rate_limit": {
      "backend": "memory",
      "redis_url": null,
      "requests_per_minute": 60,
      "requests_per_hour": 500
    },
    "metrics_enabled": true,
    "metrics_auth_required": false
  }
}
```

---

## 9. Acceptance Test Summary

| ID | Test | Pass Criteria |
|----|------|---------------|
| 3.1.1 | Function not split | Logical unit in single chunk |
| 3.1.2 | Markdown splits at headers | Chunks start near ## |
| 3.1.3 | Overlap present | ~300 chars shared between chunks |
| 3.1.4 | Large files handled | No crash on big files |
| 3.2.1 | History appends | Multiple runs create multiple entries |
| 3.2.2 | History filtering | --since and --id work |
| 3.2.3 | Webhook fires | POST received at webhook URL |
| 3.2.4 | Webhook signature | Signature validates correctly |
| 3.3.1 | Multi-context search | Results from multiple contexts |
| 3.3.2 | Results tagged | Context name on each result |
| 3.3.3 | API multi-context | Mixed results via API |
| 3.3.4 | Allowlist respected | Cannot search unauthorized contexts |
| 3.4.1 | Auto-archive | Old docs flagged automatically |
| 3.4.2 | Archived excluded | Not in default search |
| 3.4.3 | Include-archive | Old docs found with flag |
| 3.4.4 | Restore | Doc back in active search |
| 3.4.5 | Purge | Very old docs permanently removed |
| 3.5.1 | Redis rate limit | Persists across restart |
| 3.5.2 | Metrics endpoint | Prometheus format returned |
| ~~3.5.3~~ | ~~Request ID~~ | ✅ Completed in P2 |
| ~~3.5.4~~ | ~~Detailed health~~ | ✅ Completed in P2 |

---

*End of P3 spec.*
