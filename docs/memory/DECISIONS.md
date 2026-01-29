# Decisions

### 2026-01-29 — Created memory file structure

- **Why:** P4 requires persistent state tracking for digest/brief generation
- **Impact:** Claude can now load context from STATE.md on session start
- **Evidence:** `docs/memory/` directory

### 2026-01-29 — P4 implementation complete

- **Why:** Session bootstrap + daily digest needed for daily-usable memory service
- **Impact:** Claude starts with context loaded, proactive surfacing via digest/brief
- **Evidence:** `specs/P4_IMPLEMENTATION_SPEC.md`, `docs/plans/2026-01-29-p4-implementation.md`
