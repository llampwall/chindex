<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Gateway dashboard integration: accurate file/chunk counts and sync status reporting

## Active Work
- All recent changes committed and clean working tree

## Blockers
None

## Next Actions
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
Commits covered through: 1556aaf4112000488d46f31c0d93069fd7799cee

<!-- chinvex:last-commit:1556aaf4112000488d46f31c0d93069fd7799cee -->
