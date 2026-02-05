# src/chinvex/memory_templates.py
from __future__ import annotations

import datetime
from pathlib import Path


def get_state_template(commit_hash: str = "unknown") -> str:
    """Return STATE.md template with coverage anchor."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Unknown (needs human)

## Active Work
Unknown (needs human)

## Blockers
Needs triage

## Next Actions
- [ ] Review this file and update with current project state
- [ ] Fill in Quick Reference with actual commands

## Quick Reference
- Run: (command to start/run the project)
- Test: (command to run tests)
- Entry point: (main file or module)

## Out of Scope (for now)
(Nothing explicitly deferred yet)

---
Last memory update: {today}
Commits covered through: {commit_hash}

<!-- chinvex:last-commit:{commit_hash} -->
"""


def get_constraints_template() -> str:
    """Return CONSTRAINTS.md template with core sections."""
    return """<!-- DO: Add bullets. Edit existing bullets in place with (updated YYYY-MM-DD). -->
<!-- DON'T: Delete bullets. Don't write prose. Don't duplicate — search first. -->

# Constraints

## Infrastructure
- (Technical limits, batch sizes, ports, paths)

## Rules
- (Invariants, "don't do X because Y")

## Key Facts
- (Lookup values: URLs, env var names, commands)

## Hazards
- (Things that bite you if you forget)

## Superseded
(None yet)
"""


def get_decisions_template() -> str:
    """Return DECISIONS.md template with current month section."""
    current_month = datetime.datetime.now().strftime("%Y-%m")
    return f"""<!-- DO: Append new entries to current month. Rewrite Recent rollup. -->
<!-- DON'T: Edit or delete old entries. Don't log trivial changes. -->

# Decisions

## Recent (last 30 days)
- (5-10 bullet summary of recent decisions — rewritable)

## {current_month}

### YYYY-MM-DD — [Decision title]

- **Why:** [Reason for the decision]
- **Impact:** [What changed as a result]
- **Evidence:** [commit hash or PR link]

---

**Bug fix format:**

### YYYY-MM-DD — Fixed [bug description]

- **Symptom:** [What you observed]
- **Root cause:** [Why it happened]
- **Fix:** [What you did]
- **Prevention:** [How to avoid in future]
- **Evidence:** [commit hash]
"""


def bootstrap_memory_files(repo_root: Path, initial_commit_hash: str = "unknown") -> None:
    """Create docs/memory/ with STATE.md, CONSTRAINTS.md, DECISIONS.md if they don't exist.

    Args:
        repo_root: Root of the git repository
        initial_commit_hash: Starting commit hash for coverage anchor
    """
    memory_dir = repo_root / "docs" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Create STATE.md if missing
    state_file = memory_dir / "STATE.md"
    if not state_file.exists():
        state_file.write_text(get_state_template(initial_commit_hash))

    # Create CONSTRAINTS.md if missing
    constraints_file = memory_dir / "CONSTRAINTS.md"
    if not constraints_file.exists():
        constraints_file.write_text(get_constraints_template())

    # Create DECISIONS.md if missing
    decisions_file = memory_dir / "DECISIONS.md"
    if not decisions_file.exists():
        decisions_file.write_text(get_decisions_template())
