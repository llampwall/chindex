---
name: using-chinvex
description: Use when working with chinvex knowledge management system - searching knowledge base, ingesting repos, managing contexts, syncing metadata changes from strap registry, or switching embedding providers
---

# Using Chinvex

## Overview

Chinvex is a hybrid retrieval engine (SQLite FTS5 + ChromaDB vectors) for personal knowledge management. Indexes code repos and chat logs for semantic search.

**Key concept:** Contexts group related repos. One context can contain multiple repos. Sync daemon watches ALL contexts automatically.

## Quick Reference

| Task | Command |
|------|---------|
| Search all contexts | `chinvex search --all "query"` |
| Search specific context | `chinvex search -c name "query"` |
| Add new repo | `chinvex ingest -c name --repo /path` |
| Start sync daemon | `chinvex sync start` |
| Sync metadata from strap | `chinvex context sync-metadata-from-strap -c name` |
| Check status | `chinvex status` |
| Generate session brief | `chinvex brief -c name` |

## Core Workflows

### Initial Setup: New Repo

```bash
# Creates context if needed, ingests repo
chinvex ingest -c my_project --repo P:\repos\my_project --status active
```

### Searching

```bash
# Search specific context
chinvex search -c my_project "authentication error handling"

# Search everything
chinvex search --all "FastAPI middleware"

# With filters
chinvex search -c name "query" --k 10 --source repo --rerank
```

### Auto-Update: Sync Daemon (Recommended)

Sync daemon watches ALL repos across ALL contexts automatically:

```bash
# Start once - watches everything in background
chinvex sync start

# Check status
chinvex sync status

# Ensure running (safe for startup scripts)
chinvex sync ensure-running
```

Watches for file changes → debounces 30s → triggers delta ingest (only changed files).

**Manual ingest when needed:**
```bash
chinvex ingest -c name                    # Full re-ingest
chinvex ingest -c name --paths file1,file2 # Delta ingest
```

### Metadata Management: Strap Integration

**Source of truth:** `registry.json` in strap stores metadata (status, tags, depth)

**Edit strap registry.json, then sync:**

```bash
# For status/tags changes (no re-embedding needed)
chinvex context sync-metadata-from-strap -c name

# For depth changes (requires full rebuild)
chinvex context sync-metadata-from-strap -c name
chinvex ingest -c name --rebuild-index
```

### Switching Embedding Providers

**CRITICAL: Use `--rebuild-index`, NOT purge+ingest**

```bash
# Correct: Preserves chunks, only re-embeds
chinvex ingest -c name --embed-provider ollama --rebuild-index
```

**DO NOT purge then ingest:**
```bash
# ❌ WRONG: Wastes work by re-chunking everything
chinvex context purge name
chinvex ingest -c name --embed-provider ollama
```

**Why --rebuild-index is required:**
- Dimension mismatch (OpenAI: 1536 dims, Ollama: varies) → query failures without rebuild
- `--rebuild-index` preserves chunks, only re-embeds (efficient)
- Purge+ingest re-chunks everything (wasteful, slower)

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Switch providers without `--rebuild-index` | Always include `--rebuild-index` with `--embed-provider` |
| Purge+ingest when switching providers | Use `ingest --rebuild-index` instead (preserves chunks) |
| Depth change without sync-metadata | Run `sync-metadata-from-strap` first, then `ingest --rebuild-index` |
| Status/tags change with full rebuild | Just `sync-metadata-from-strap` (no ingest needed) |
| Edit context.json directly for metadata | Edit `registry.json` in strap, then sync-metadata-from-strap |
| Manual re-ingest every change | Use `sync start` - handles updates automatically |

## Two-Step Workflows

### Depth Change (shallow → deep or vice versa)
```bash
# Step 1: Sync metadata from strap
chinvex context sync-metadata-from-strap -c name

# Step 2: Rebuild index with new depth
chinvex ingest -c name --rebuild-index
```

### Provider Switch (OpenAI ↔ Ollama)
```bash
# Single command - rebuild required, preserves chunks
chinvex ingest -c name --embed-provider ollama --rebuild-index

# ❌ DO NOT: purge then ingest (wasteful)
# chinvex context purge name && chinvex ingest -c name --embed-provider ollama
```

### Status/Tags Change (no rebuild needed)
```bash
# Single command - just metadata sync
chinvex context sync-metadata-from-strap -c name
```

## Memory System

Generate briefs and update memory files:

```bash
chinvex brief -c name                      # Session brief (loaded by hooks)
chinvex update-memory -c name              # Review mode (shows diff)
chinvex update-memory -c name --commit     # Auto-commit changes
```

Memory files: `docs/memory/{STATE,CONSTRAINTS,DECISIONS}.md`

## Command Reference

### ingest - Add/update content
```bash
chinvex ingest -c name [OPTIONS]
  --repo PATH              Add repo (repeatable)
  --chat-root PATH         Add chat logs
  --embed-provider TEXT    openai|ollama [default: openai]
  --rebuild-index          Wipe and re-embed everything
  --chinvex-depth TEXT     full|light|index [default: full]
  --status TEXT            active|stable|dormant
  --tags TEXT              Comma-separated tags
  --paths TEXT             Delta ingest (comma-separated files)
  --register-only          Register without ingesting
  --rechunk-only           Rechunk, reuse embeddings
```

### search - Query knowledge base
```bash
chinvex search "query" [OPTIONS]
  -c, --context TEXT       Search specific context
  --contexts TEXT          Comma-separated contexts
  --all                    Search all contexts
  --exclude TEXT           Exclude contexts (with --all)
  --k INTEGER              Top K results [default: 8]
  --min-score FLOAT        Score threshold [default: 0.35]
  --source TEXT            all|repo|chat|codex_session
  --rerank                 Enable reranking
```

### sync - File watcher daemon
```bash
chinvex sync start               # Start watching all contexts
chinvex sync stop                # Stop daemon
chinvex sync status              # Check status/heartbeat
chinvex sync ensure-running      # Idempotent start
chinvex sync reconcile-sources   # Update watched sources
```

### context - Context management
```bash
chinvex context create NAME
chinvex context list
chinvex context sync-metadata-from-strap -c NAME
chinvex context purge NAME       # Delete one context
chinvex context purge --all      # Delete ALL contexts
chinvex context rename OLD NEW
chinvex context remove-repo -c NAME --repo PATH
```

### Other commands
```bash
chinvex status                    # All contexts status
chinvex brief -c NAME             # Session brief
chinvex update-memory -c NAME     # Update memory files from git
chinvex eval -c NAME              # Run evaluation suite
chinvex gateway serve             # HTTP API server
chinvex hook install              # Git post-commit hook
```

## Configuration

**Contexts root:** `P:\ai_memory\contexts\` or `CHINVEX_CONTEXTS_ROOT` env var

**Context structure:**
```
contexts/name/
  context.json          # Config (includes, embeddings)
  hybrid.db             # SQLite FTS5 index
  meta.json             # Provider/model/dimensions
  chroma/               # Vector embeddings
  .chinvex-status.json  # Daemon status (PID)
```

## Real-World Tips

- **Sync daemon for active work** - set once, forget
- **Search --all for broad queries** - crosses all contexts
- **Rebuild only when required** - provider/depth changes only
- **Strap is source of truth** - sync-metadata after editing
- **Delta ingest is efficient** - sync daemon uses it automatically
