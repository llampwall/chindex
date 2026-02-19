<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Test suite stabilized (63 failures fixed); next: eval validation and P5b planning

## Active Work
- Clean working tree; all recent changes committed

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
Last memory update: 2026-02-19
Commits covered through: 1de51147f0db2912cd873ed357541406bc36add6

<!-- chinvex:last-commit:1de51147f0db2912cd873ed357541406bc36add6 -->
