<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Skills, backup infrastructure, and embedding provider hardening

## Active Work
- Completed proper connection management for ChromaDB and SQLite
  - VectorStore.close() method using ChromaDB's _system.stop()
  - Context manager support for automatic cleanup
  - Gateway shutdown handler for graceful connection cleanup
  - Fixed Windows file lock issues (PermissionError on database deletion)
- Completed using-chinvex skill for Claude Code and Codex with comprehensive CLI workflow docs
- Completed automatic context.json backup system (30 backups, auto-prune)
- Completed OpenAI as default embedding provider; search reads provider from meta.json

## Blockers
None

## Next Actions
- [ ] Test dashboard status integration end-to-end
- [ ] Validate depth change workflow (sync metadata + rebuild-index)
- [ ] Complete P5b planning and implementation (memory maintainer, startup hooks)
- [ ] Validate eval suite with >=80% hit rate baseline

## Quick Reference
- Install: `pip install -e .` (requires Python 3.12, venv)
- Ingest: `chinvex ingest --context <name> --repo <path>`
- Search: `chinvex search --context <name> "query"`
- Sync metadata: `chinvex context sync-metadata-from-strap --context <name>`
- Test: `pytest`
- Entry point: `src/chinvex/cli.py`

## Out of Scope (for now)
- Scheduled memory maintenance (deferred to P6)
- Cross-context search UI improvements
- Automated golden query generation

---
Last memory update: 2026-02-16
Commits covered through: 5adfbfa71318cc84b7524f34595fe56c9432295c

<!-- chinvex:last-commit:5adfbfa71318cc84b7524f34595fe56c9432295c -->
