# P4 Implementation Spec

**Headline: Session bootstrap + daily digest**

## Goal

Turn Chinvex into a daily-usable memory service with:
- Stable tool surface (Gateway + MCP) ✓ DONE
- Proactive surfacing (digest + brief artifacts)
- Pluggable embeddings (Ollama ⇄ OpenAI)
- Session bootstrap (Claude starts with context loaded)

## Non-goals (P5+)

- Cron-based nudge agent / scheduling intelligence
- Multi-user auth
- UI beyond CLI + LLM clients

---

## P4.0 MCP Server ✓ COMPLETE

Claude Code can query Chinvex via MCP with same power as ChatGPT Actions.

---

## P4.1 Inline Context Creation

### Outcome

`chinvex ingest` auto-creates context if it doesn't exist.

### Implementation

```bash
chinvex ingest --context NewProject --repo C:\Code\newproject
```

If `NewProject` doesn't exist in registry:
1. Create `contexts/NewProject/context.json` with sensible defaults
2. Create index directories
3. Proceed with ingest

### CLI changes

```
chinvex ingest --context <n> [--repo <path>]... [--chat-root <path>]... [--no-write-context]
```

- `--repo` and `--chat-root` append **only if not already present** (dedupe by normalized absolute path)
- `--no-write-context` ingests ad-hoc without mutating context.json
- First `--repo` becomes primary if creating new context

### Path normalization

For deduplication:
- Convert to absolute path
- Use forward slashes (even on Windows)
- Case-insensitive comparison on Windows
- Example: `.\src` and `C:\Code\chinvex\src` resolve to same normalized path

### Error handling

- `--repo` path doesn't exist → **fail immediately** with clear error (don't silently skip)
- Context exists but index directories missing → recreate them
- `--no-write-context` on non-existent context → create context in memory, ingest, but don't persist context.json

### Acceptance

- `chinvex ingest --context Foo --repo ./bar` on fresh context creates and ingests
- Existing contexts just ingest (no config clobber)
- Duplicate `--repo` paths are ignored (no accumulation)

---

## P4.2 Embedding Provider Abstraction

### Outcome

Ingest works with Ollama OR OpenAI embeddings, switchable per-context.

### Interface

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
```

### Providers

| Provider | Model | Dims | Cost |
|----------|-------|------|------|
| Ollama | mxbai-embed-large | 1024 | Free (self-hosted) |
| OpenAI | text-embedding-3-small | 1536 | ~$0.02/1M tokens |

### Selection precedence

1. CLI: `--embed-provider ollama|openai`
2. context.json: `"embedding": {"provider": "openai", "model": "text-embedding-3-small"}`
3. Env: `CHINVEX_EMBED_PROVIDER`
4. Default: `ollama`

### API key handling

- OpenAI: `OPENAI_API_KEY` environment variable (standard OpenAI convention)
- Ollama: no key required (local or remote host)

### Provider-specific batching

| Provider | Max batch (count) | Max batch (bytes) | Timeout | Retry |
|----------|-------------------|-------------------|---------|-------|
| Ollama | 64 texts | 1 MB | 60s | 3x exponential backoff |
| OpenAI | 2048 texts | 1 MB | 30s | 3x exponential backoff |

Both caps enforced (whichever triggers first).

### Failure behavior

- **Atomic ingest**: If embedding fails mid-ingest, fail the entire operation (no partial writes)
- Provider unavailable → fail with clear error, suggest fallback provider
- Rate limit hit → retry with backoff, fail after 3 attempts

### Dimension safety

**Source of truth: index metadata** (not context.json).

On ingest:
1. Read index metadata (`indexes/X/meta.json`) for current dims/provider
2. If missing, initialize from provider
3. If present, enforce match or fail

context.json stores **intent**; index stores **reality**.

```json
// indexes/X/meta.json
{
  "schema_version": 2,
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "embedding_dimensions": 1536,
  "created_at": "2026-01-29T12:00:00Z"
}
```

Refuse mismatched embeddings unless `--rebuild-index`.

### `--rebuild-index` behavior

When invoked:
1. **Wipe** existing index (SQLite FTS + Chroma collection)
2. **Rewrite** `meta.json` completely (no merge with old values)
3. **Re-embed** all docs from source files
4. **Preserve** watch history (watches are queries, not tied to embeddings)

This is destructive and slow. Use only when switching providers.

### Acceptance

- `chinvex ingest --context X --embed-provider openai` works
- Mixing providers on same index fails with clear error
- OpenAI ingest completes in <5 min for Streamside-sized repo

---

## P4.3 Digest Generation

### Outcome

Deterministic aggregation of recent activity into structured artifact. No LLM.

### Command

```bash
chinvex digest --context X [--since 24h] [--date YYYY-MM-DD] [--push ntfy]
```

### Inputs

- Watch history (JSONL) since last digest or `--since`
- Ingest run log (see below)
- `docs/memory/STATE.md` if present (repo-level state)
- Previous digest (most recent by filename `YYYY-MM-DD.md`)

### Ingest run log (required for stable deltas)

Append-only `contexts/X/ingest_runs.jsonl`. Two records per run to handle crashes:

**Run start:**
```json
{"run_id": "550e8400-...", "status": "started", "started_at": "2026-01-29T12:00:00Z", "sources": [...]}
```

**Run end (success):**
```json
{"run_id": "550e8400-...", "status": "succeeded", "ended_at": "2026-01-29T12:05:00Z", "docs_seen": 289, "docs_changed": 12, "chunks_new": 47, "chunks_updated": 103}
```

**Run end (failure):**
```json
{"run_id": "550e8400-...", "status": "failed", "ended_at": "2026-01-29T12:03:00Z", "error": "OpenAI rate limit exceeded"}
```

Digest ignores `started` records without matching `succeeded` record.

### Output

`contexts/X/digests/YYYY-MM-DD.md`:

**Timezone handling:**
- Filenames use **local date** (system timezone)
- Timestamps inside JSON use **UTC** with `Z` suffix
- `--since 24h` means 24 hours from now (not midnight-to-midnight)

```markdown
# Digest: 2026-01-29

## Watch Hits (3)
- **"retry logic"** hit 2x in `src/chinvex/embed.py` (chunks: abc123, def456)
- **"grounding"** hit 1x in `docs/P3_SPEC.md`

## Recent Changes (since last digest)
- 12 files ingested, 847 chunks updated
- Notable: `src/chinvex/gateway/evidence.py` (new)

## State Summary
- P3: COMPLETE
- P4: IN PROGRESS (MCP done, digest WIP)

## Pending Watches
- "memory consolidation" (0 hits, 3 days old)
```

Also emit `YYYY-MM-DD.json` for programmatic consumption.

### Push (optional)

`--push ntfy` sends to configured ntfy topic.

**Configuration:**
- Topic: `CHINVEX_NTFY_TOPIC` environment variable (e.g., `chinvex-alerts`)
- Server: `CHINVEX_NTFY_SERVER` env var (default: `https://ntfy.sh`)

**Keep it dumb** - no paths, no context names, no sensitive info:
```
Chinvex digest ready: 3 watch hits, 12 files changed
```

### Acceptance

- `chinvex digest --context Chinvex --since 24h` produces valid markdown
- Output is deterministic (same input → same output, except for `Generated:` timestamp)
- `--push ntfy` delivers notification
- `--date 2026-01-28` generates digest for that specific day (using historical data)

---

## P4.4 Operating Brief

### Outcome

Session-start artifact that loads context for both human AND Claude.

### Command

```bash
chinvex brief --context X [--output SESSION_BRIEF.md] [--repo-root <path>]
```

### Input search path (explicit order)

Brief consumes files in this order (missing files skipped silently, section omitted from output):

1. `{repo_root}/docs/memory/STATE.md` (full content)
2. `{repo_root}/docs/memory/CONSTRAINTS.md` (until first `## ` header only)
3. `{repo_root}/docs/memory/DECISIONS.md` (entries from last 7 days, parsed by `### YYYY-MM-DD` headings)
4. `contexts/X/digests/YYYY-MM-DD.md` (most recent by filename)
5. Watch history (last 5 hits OR last 24h, whichever yields more)

**Repo root detection:**
- If `--repo-root` specified, use that
- Else walk up from CWD looking for `.git/` or `docs/memory/`
- Escape hatch for monorepos, subdirs, worktrees

### Output

`SESSION_BRIEF.md` (or stdout):

```markdown
# Session Brief: Chinvex
Generated: 2026-01-29T14:30:00

## Current State
P4 in progress. MCP server complete. Working on digest/brief.

## Active Work
- Implementing P4.3 digest generation
- Testing OpenAI embedding provider

## Blockers
None

## Next Actions
- [ ] Complete P4.3 digest implementation
- [ ] Wire startup hook for Claude Code

## Constraints (highlights)
- Embedding dims locked per index
- ChromaDB batch limit: 5000 vectors
- Metrics endpoint requires auth

## Recent Decisions (7d)
- 2026-01-29: MCP server uses HTTP client to gateway (not direct index access)
- 2026-01-28: contexts="all" accepted as string in gateway validation

## Recent Activity (24h)
- 3 watch hits (retry logic, grounding, embeddings)
- 12 files changed in last ingest

## Context Files
- State: `docs/memory/STATE.md`
- Digest: `contexts/Chinvex/digests/2026-01-29.md`
```

### Mode: deterministic vs ranked

- Default: deterministic template (no LLM)
- `--rank`: optional LLM pass to reorder by urgency/ROI (future)

### Acceptance

- `chinvex brief --context Chinvex` produces valid markdown
- Output includes state, recent activity, constraints, recent decisions
- Works without any memory files (graceful degradation - sections just omitted)
- Missing `docs/memory/` directory → warning but still outputs digest/watch sections

---

## P4.5 Startup Hook

### Outcome

When Claude Code opens a repo, it automatically has context loaded.

### Implementation

**Option B is the baseline** (CLAUDE.md/AGENTS.md instruction):

```markdown
## Session Start Protocol
On session start, run: `chinvex brief --context <n>`
Read the output before proceeding with any work.
```

**Option A is nice-to-have** (if Claude Code supports hooks):
```json
{
  "onSessionStart": "chinvex brief --context ${REPO_NAME} --output .claude/SESSION_BRIEF.md"
}
```

**Option C (future)**: MCP resource `chinvex://brief/X` that Claude auto-fetches.

### Acceptance

- Opening Claude Code in chinvex repo surfaces the brief (via Option B minimum)
- Claude's first response demonstrates awareness of current state

---

## P4.6 Observability

### Metrics (extend existing Prometheus)

```
chinvex_embeddings_total{provider="openai|ollama"}
chinvex_embeddings_latency_seconds{provider}
chinvex_embeddings_retries_total{provider}
chinvex_digest_generated_total{context}
chinvex_brief_generated_total{context}
chinvex_ingest_runs_total{context}
```

### Logging

- Embedding batches: count, bytes, latency
- Digest generation: inputs found, output size
- Brief generation: sections included
- Ingest runs: docs seen/changed, chunks new/updated

---

## P4.7 Runbooks

### start_gateway.ps1 (exists)

### start_mcp.ps1 (new)
```powershell
$env:CHINVEX_API_TOKEN = (Get-Content ~/.secrets/chinvex_token)
chinvex-mcp
```

### backup.ps1 (new)
```powershell
# Snapshot context registry + indexes + digests
$timestamp = Get-Date -Format "yyyy-MM-dd"
$dest = "P:\backups\chinvex\$timestamp"
Copy-Item -Recurse P:\ai_memory\contexts $dest\contexts
Copy-Item -Recurse P:\ai_memory\indexes $dest\indexes
```

---

## P4.8 Tests

### Unit

- [ ] Embedding provider selection precedence
- [ ] OpenAI provider batching + retry (mock with `responses` library)
- [ ] Digest generation determinism (golden file in `tests/fixtures/`)
- [ ] Brief generation with missing files (graceful degradation)
- [ ] Repo path deduplication and normalization
- [ ] DECISIONS.md date parsing

### E2E

- [ ] `chinvex ingest --context New --repo ./test` creates context
- [ ] `chinvex digest --context X` produces valid output
- [ ] `chinvex brief --context X` produces valid output
- [ ] MCP `chinvex_search` returns grounded results (mock stdio)
- [ ] `--rebuild-index` wipes and regenerates

---

## Dependencies

- **P3 required**: P4 builds on P3 (cross-context search, watch history, webhooks, metrics)
- **Watch history optional for digest**: If no watches exist, digest just shows ingest activity
- **Gateway required for MCP**: MCP server calls gateway HTTP API, won't work without it
- **Gateway hot-reload**: Gateway picks up new contexts on next request (no restart needed)

---

## Implementation Order

0. **Setup** Create `docs/memory/` with STATE.md, CONSTRAINTS.md, DECISIONS.md (one-time)
1. **P4.1** Inline context creation (quick, unblocks testing)
2. **P4.2** Embedding provider abstraction (unblocks fast ingest)
3. **P4.3** Digest generation (deterministic, testable)
4. **P4.4** Brief generation (consumes digest + memory files)
5. **P4.5** Startup hook (wires it together)
6. **P4.6** Observability (extend existing)
7. **P4.7** Runbooks (quick win)
8. **P4.8** Tests (parallel with implementation)

---

## Migration

- context.json schema stays v1 (embedding field is optional addition)
- Index `meta.json` is new:
  - **First ingest on existing index**: read Chroma collection metadata to infer dims, create `meta.json`
  - **First ingest on new index**: create `meta.json` from provider
  - **Inference fails** (empty collection, no metadata): prompt user to specify `--embed-provider` or fail
- Auto-upgrade existing indexes: infer dims from existing Chroma collection
- No forced reindex unless provider change

---

## Deferred to P5

- Cron-based nudge agent
- Intelligent reminder scheduling
- Cross-context summarization

---

## Appendix: Memory File Format

### Structure

```
{repo_root}/docs/memory/
├── STATE.md        # Rewrite allowed
├── CONSTRAINTS.md  # Stable sections, bullets only
└── DECISIONS.md    # Append-only
```

**Location:** Always `{repo_root}/docs/memory/` (not per-context, per-repo).

### STATE.md (rewrite allowed)

The "load me into Claude's head" file. Claude can completely replace content, but should preserve section structure.

**Template:**
```markdown
# State

## Current Objective
P4 implementation - session bootstrap + daily digest

## Active Work
- Implementing digest generation
- Testing OpenAI embeddings

## Blockers
None

## Next Actions
- [ ] Complete P4.3 digest CLI
- [ ] Wire startup hook
```

Keep it short. If it grows past ~30 lines, move details elsewhere.

### CONSTRAINTS.md (stable sections)

Replaces ADRs + key_facts + hazards. Section headers are stable (don't rename), but bullets can be added/updated/removed.

**Template:**
```markdown
# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors
- Embedding dims locked per index (see meta.json)
- Gateway port: 7778

## Rules
- Schema stays v2 - no migrations without rebuild
- Metrics endpoint requires auth
- Archive is dry-run by default

## Key Facts
- Gateway: localhost:7778 → chinvex.unkndlabs.com
- Contexts root: P:\ai_memory\contexts
- Token env var: CHINVEX_API_TOKEN
```

No ADR-001 ceremony. Just facts. Categories are extensible (add new `## Section` as needed).

### DECISIONS.md (append-only)

Replaces worklog + bugs. **Append-only**: entries can be edited for typos but not deleted.

**Template:**
```markdown
# Decisions

### 2026-01-29 — MCP server uses HTTP client to gateway

- **Why:** Allows Claude Code to query from any machine, not just where index lives
- **Impact:** MCP server depends on gateway being up; no direct index access
- **Evidence:** `src/chinvex_mcp/server.py`

### 2026-01-28 — contexts="all" accepted as string in gateway

- **Why:** OpenAPI spec promised it, validation was rejecting it
- **Impact:** Cross-context search works from ChatGPT Actions
- **Evidence:** commit abc1234
```

### Maintainer Trigger Rules

| File | Trigger | Action |
|------|---------|--------|
| STATE.md | Every commit / session end | Rewrite |
| CONSTRAINTS.md | "Learned something the hard way" | Add/update bullet |
| DECISIONS.md | "Something became true because of a change" | Append entry |

**Who is the maintainer?** Either:
- Claude (via post-commit hook or session-end protocol)
- User (manual updates)
- Future: automated agent

The DECISIONS trigger is key: not "when should I write an ADR?" but "did something just become true?"
