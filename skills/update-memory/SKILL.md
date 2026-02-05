---
name: update-memory
description: Update project memory files (STATE/CONSTRAINTS/DECISIONS) from git history using LLM analysis
---

# Update Memory

## Overview

Analyzes git commits since last memory update and regenerates STATE.md, CONSTRAINTS.md, and DECISIONS.md.

**You are the LLM inference layer.** Read git history → analyze changes → update memory files respecting their update modes.

## Usage

```
/update-memory
```

Run from within a repository that has `docs/memory/` files.

## Step 1: Extract Last Processed Commit

Read the coverage anchor from STATE.md:

```bash
grep "chinvex:last-commit" docs/memory/STATE.md
```

Extract the commit hash. If not found, this is the first run - process full history.

## Step 2: Check if Files Are Bootstrap Templates

Check if STATE.md still has bootstrap markers:

```bash
grep -E "Unknown \(needs human\)|Review this file and update" docs/memory/STATE.md
```

**If bootstrap markers found:** This is effectively a first run - ignore the footer hash and process recent history.

## Step 3: Get Git History

Get commits since last update (or last 50 if first run/bootstrap):

```bash
# If hash found AND no bootstrap markers:
git log <hash>..HEAD --format="%H%n%an%n%aI%n%s%n%b%n---END---" --stat

# If first run OR bootstrap templates detected:
git log -50 --format="%H%n%an%n%aI%n%s%n%b%n---END---" --stat
```

**If no new commits AND files are already populated (not bootstrap):** Tell user memory files are up to date and stop.

## Step 4: Get Current HEAD

```bash
git rev-parse HEAD
```

Save this - you'll need it for the footer update.

## Step 5: Read Current Memory Files

```bash
cat docs/memory/STATE.md
cat docs/memory/CONSTRAINTS.md
cat docs/memory/DECISIONS.md
```

## Step 6: Read the Spec

```bash
cat P:\software\chinvex\docs\PROJECT_MEMORY_SPEC_v0.3.md
```

**Critical sections:**
- STATE.md: Full rewrite, use "Unknown (needs human)" when uncertain, 45 line hard cap
- CONSTRAINTS.md: Merge-only, dedupe before adding, never delete
- DECISIONS.md: Append entries + rewrite Recent rollup

## Step 7: Analyze and Generate Updates

**Analyze the git history you gathered.** Look for:

**STATE.md:**
- Current objective (from specs, commit patterns, latest work)
- Active work (what's in progress based on recent commits)
- Blockers (mentioned in commits or visible from stalled work)
- Next actions (logical next steps from current state)
- Quick Reference (if you can determine run/test/entry commands from codebase)
- Use "Unknown (needs human)" / "Needs triage" when you can't infer with confidence

**CONSTRAINTS.md:**
- New infrastructure facts (ports, paths, limits discovered in commits)
- New rules ("don't do X because Y" learned from bugs/issues)
- New hazards (things that bit us, mentioned in fixes)
- **Search existing bullets first** - if similar exists, update it in place with `(updated YYYY-MM-DD)`
- Never delete - move obsolete to Superseded

**DECISIONS.md:**
- Apply logging threshold: interfaces, storage, workflow, security, performance, dependencies, user-facing changes
- Create dated entries (### YYYY-MM-DD — Title) in current month section
- Use bug-fix format for bugs (Symptom/Root cause/Fix/Prevention)
- Include commit hash as Evidence
- Update Recent rollup (5-10 bullet summary of last 30 days)

## Step 8: Write Updated Files

Use the Write tool to write each file:

```
Write docs/memory/STATE.md with:
<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
[your analysis]

## Active Work
[your analysis]

## Blockers
[your analysis]

## Next Actions
[your analysis]

## Quick Reference
[if you can determine it]

## Out of Scope (for now)
[if applicable]

---
Last memory update: [TODAY]
Commits covered through: [HEAD hash from Step 3]

<!-- chinvex:last-commit:[HEAD hash] -->
```

Repeat for CONSTRAINTS.md and DECISIONS.md, respecting their update modes.

## Step 9: Show Changes and Commit

```bash
git diff docs/memory/
```

Ask user: "Memory files updated. Review the changes above. Commit? [Y/n]"

If yes:
```bash
git add docs/memory/
git commit -m "docs: update memory files

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

## Key Rules

- **Never hallucinate** - use "Unknown (needs human)" when uncertain
- **STATE hard cap: 45 lines** - be concise
- **CONSTRAINTS dedupe** - search before adding new bullets
- **DECISIONS need evidence** - every entry needs commit hash
- **Respect update modes** - rewrite/merge/append per spec
