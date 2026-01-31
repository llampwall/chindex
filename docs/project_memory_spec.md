# Project Memory System - Spec Draft v0.2

> **Status:** Draft for final review
> **Purpose:** Define the 3-file project memory system and its maintenance
> **Changes:** v0.2 incorporates ChatGPT feedback (coverage anchor, superseded mechanics, rollups, bug format, scope boundaries, section rules)

---

## Problem Statement

The original 5-file system (`operating_brief.md`, `key_facts.md`, `adrs.md`, `bugs.md`, `worklog.md`) failed because:

1. **ADRs never got updated** - format too ceremonial, agents couldn't decide what qualified
2. **Multiple files competed** - agents spread updates thin, missed important ones
3. **Unclear triggers** - "when should I write an ADR?" has no crisp answer
4. **Too much structure** - high risk of rewriting history, so agents avoided touching files

The 3-file system fixes this by:
- Consolidating related concerns (key_facts + adrs + hazards → CONSTRAINTS)
- Clear update modes (rewrite vs merge vs append)
- Trigger based on "did something become true?" not "should I write an ADR?"

---

## File Structure

```
{repo_root}/docs/memory/
├── STATE.md        # Rewrite allowed
├── CONSTRAINTS.md  # Merge-only (add/update, don't delete)
└── DECISIONS.md    # Append-only
```

**Location:** Always `{repo_root}/docs/memory/` (per-repo, not per-context).

---

## File Specifications

### STATE.md

**Purpose:** "Load me into Claude's head" - what's true right now.

**Update mode:** Full rewrite allowed. This is the only file that should feel "current."

**Required sections:**
```markdown
# State

## Current Objective
[One line: what we're trying to accomplish]

## Active Work
- [Bullet list of in-progress items]

## Blockers
[None, or bullet list]

## Next Actions
- [ ] [Checkbox list of immediate next steps]

## Out of Scope (for now)
- [Things explicitly deferred to prevent initiative creep]

---
Last memory update: YYYY-MM-DD
Commits covered through: <hash>
```

**Rules:**
- Keep it SHORT. If it grows past ~30 lines, move details elsewhere.
- No history. No rationale. Just current truth.
- Update every session or when objective changes.
- **Coverage anchor required:** Footer must include last update date and ending commit hash. Maintainer processes `git log <last_hash>..HEAD` then updates footer with new ending hash.

---

### CONSTRAINTS.md

**Purpose:** What must not change - rules, hazards, infrastructure facts that constrain future work.

**Update mode:** Merge-only. Add/update bullets, don't delete unless explicitly told.

**Sections:**
```markdown
# Constraints

## Infrastructure
- [Technical limits, batch sizes, ports, paths]

## Rules
- [Invariants, "don't do X because Y"]

## Key Facts
- [Lookup values: URLs, env var names, commands]

## Hazards
- [Things that bite you if you forget]

## Superseded
- (Superseded YYYY-MM-DD) [Old constraint that no longer applies, with reason]
```

**Rules:**
- Bullets only. No prose.
- **Core sections (always present):** Infrastructure, Rules, Key Facts, Hazards, Superseded
- **Optional sections (add as needed):** APIs, Performance, Security, Dependencies
- Trigger: "learned something the hard way"
- **Supersede, don't delete:** When a constraint no longer applies, move it to `## Superseded` with date and reason. Never delete outright - preserves history while keeping hot path scannable.

---

### DECISIONS.md

**Purpose:** Audit trail - how did we get here?

**Update mode:** Append-only for entries. Top rollup section can be rewritten.

**Structure:**
```markdown
# Decisions

## Recent (last 30 days)
- [5-10 bullet summary of recent decisions - rewritable]

## 2026-01
[Monthly sections for append-only entries]

### YYYY-MM-DD — [Decision title]

- **Why:** [Reason for the decision]
- **Impact:** [What changed as a result]
- **Evidence:** [commit hash / file / link]
```

**Bug fix entries** (when decision is a bug fix, use this sub-shape):
```markdown
### YYYY-MM-DD — Fixed [bug description]

- **Symptom:** [What you observed]
- **Root cause:** [Why it happened]
- **Fix:** [What you did]
- **Prevention:** [How to avoid in future]
- **Evidence:** [commit hash]
```

**Rules:**
- Trigger: "something became true because of a change"
- Captures: architectural choices, bug resolutions, lessons learned
- No ADR-001 numbering. Dates only.
- **Monthly sections:** Organize by `## YYYY-MM` to keep inserts localized
- **Rollup section:** Top "Recent (last 30 days)" is rewritable for quick scanning
- **Bug playbook preserved:** Bug fixes use extended format so playbook value isn't lost
- Recent entries (last 7 days) feed into session briefs.

---

## Maintainer System

### Trigger: Scheduled Batch (NOT per-commit)

**Rationale:**
- Claude/Codex commit frequently - per-commit hooks are too expensive
- Batch processing lets agent see patterns across multiple commits
- Can be triggered on-demand when needed
- Requires good commit messages and specs/plans as source material

**Options:**

| Mode | Trigger | Use case |
|------|---------|----------|
| Scheduled | Daily/twice-daily cron | Hands-off maintenance |
| On-demand | `chinvex update-memory --context X` | Before starting work |
| Session-end | Manual or hook | After significant work sessions |

### Input Sources (for maintainer agent)

1. **Git log** - commits since last update
2. **Specs/plans** - `specs/*.md`, `docs/plans/*.md`
3. **Current memory files** - to avoid duplicating existing content
4. **Diff of changed files** - to understand what actually changed

### Update Logic

```
For each update run:
1. Read STATE.md footer to get last processed commit hash
2. Gather commits: git log <last_hash>..HEAD
3. Capture current HEAD hash for footer update
4. Read current STATE.md, CONSTRAINTS.md, DECISIONS.md
5. Read any referenced specs/plans from commits
6. For STATE.md:
   - Rewrite based on current project state
   - Use latest specs/plans to determine objective
   - **If no objective is discoverable, write: "Current Objective: Unknown (needs human)"**
   - Update footer with new date and ending commit hash
7. For CONSTRAINTS.md:
   - Check if any commits reveal new constraints
   - Merge new bullets into appropriate sections
   - Move obsolete constraints to Superseded section (never delete)
8. For DECISIONS.md:
   - Check if any commits represent significant decisions
   - Append new entries to current month section
   - Update Recent rollup with last 30 days summary
   - Trigger question: "did something become true because of a change?"
   - Use bug-fix format when applicable
```

---

## Integration with Chinvex

### Brief Generation

`chinvex brief --context X` assembles:
1. Full content of `STATE.md`
2. `CONSTRAINTS.md`: Infrastructure + Rules + Hazards sections only (skip Key Facts, Superseded)
3. `DECISIONS.md`: Recent rollup section + entries from last 7 days
4. Latest digest (watch hits, recent activity)

### Digest Generation

`chinvex digest --context X` produces daily delta:
- Watch hits
- Files changed
- Ingest stats

**Digest does NOT replace memory files** - it's a delta, not the canon.

---

## Migration from 5-file System

For repos using the old `docs/project_notes/` structure:

| Old file | Maps to | Notes |
|----------|---------|-------|
| `operating_brief.md` | `STATE.md` | Extract current state sections |
| `key_facts.md` | `CONSTRAINTS.md` | Merge into Key Facts section |
| `adrs.md` | `CONSTRAINTS.md` + `DECISIONS.md` | Rules → CONSTRAINTS, history → DECISIONS |
| `bugs.md` | `DECISIONS.md` | Each bug fix = decision entry |
| `worklog.md` | `DECISIONS.md` | Significant entries only |

---

## Open Questions

1. **Maintainer implementation:** Standalone script? Chinvex subcommand? Codex skill?

2. **Cross-repo consistency:** Should the project-context skill be updated to emit 3-file format, or is this Chinvex-specific?

3. **Frequency:** Daily? Twice daily? What time?

4. **Notification:** Should maintainer push a summary to ntfy when it runs?

5. **Validation:** How do we know the memory files are accurate/useful?

---

## Acceptance Criteria

- [ ] STATE.md reflects current project state after each maintainer run
- [ ] CONSTRAINTS.md accumulates learnings without losing history
- [ ] DECISIONS.md provides traceable audit trail
- [ ] Brief generation produces useful session-start context
- [ ] Maintainer doesn't block active development work
- [ ] Files stay short and scannable (STATE < 30 lines, CONSTRAINTS < 100 lines)

---

## References

- P4 Implementation Spec (memory file format appendix)
- Original 5-file project-context skill
- Claude/ChatGPT discussion on 3-file consolidation
