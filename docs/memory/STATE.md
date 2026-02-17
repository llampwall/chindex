<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Embedding provider migration: purge ollama defaults, establish openai as default everywhere

## Active Work
- Ollama→OpenAI default migration complete (code, tests, data, README all updated, uncommitted)

## Blockers
None

## Next Actions
- [ ] Commit ollama→openai migration changes
- [ ] Re-ingest contexts with 0 chunks using OpenAI embeddings
- [ ] Test strap uninstall end-to-end to confirm purge leaves nothing behind
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
Last memory update: 2026-02-17
Commits covered through: 7f676ea30d7254bccde40dcbd09acfec8a41d5ef

<!-- chinvex:last-commit:7f676ea30d7254bccde40dcbd09acfec8a41d5ef -->
