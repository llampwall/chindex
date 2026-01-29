# Decisions

### 2026-01-29 — Created memory file structure

- **Why:** P4 requires persistent state tracking for digest/brief generation
- **Impact:** Claude can now load context from STATE.md on session start
- **Evidence:** `docs/memory/` directory
