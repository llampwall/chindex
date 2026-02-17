<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Infrastructure hardening: connection management, purge correctness, strap uninstall cleanup

## Active Work
- All session tasks complete (purge bugs fixed, orphaned indexes cleared, scripts organized)

## Blockers
None

## Next Actions
- [ ] Test strap uninstall end-to-end to confirm purge now leaves nothing behind
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
Last memory update: 2026-02-17
Commits covered through: a074d0f9b40468ff5e1796424069d537054c796c

<!-- chinvex:last-commit:a074d0f9b40468ff5e1796424069d537054c796c -->
