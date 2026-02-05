# CLAUDE.md

## Project

[NAME] - [BRIEF_DESCRIPTION]

## Language

[LANGUAGE]

## Structure

- [RELATIVE_PATH] - [DESCRIPTION]
- [RELATIVE_PATH] - [DESCRIPTION]
...

## Commands (PowerShell)

[RUN_COMMAND]                    # [DESCRIPTION]
[COMMAND] --[FLAG]               # [DESCRIPTION]
...

## Current Sprint

See `/specs/` for implementation specs. Look at the highest phase number (P0, P1, etc.) for current work.

## Architecture

- **[COMPONENT]**: [DESCRIPTION]
- **[COMPONENT]**: [DESCRIPTION]
...

## Memory System

Chinvex repos use structured memory files in `docs/memory/`:

- **STATE.md**: Current objective, active work, blockers, next actions
- **CONSTRAINTS.md**: Infrastructure facts, rules, hazards (merge-only)
- **DECISIONS.md**: Append-only decision log with dated entries

**SessionStart Integration**: When you open a chinvex-managed repo, a hook runs `chinvex brief --context <name>` to load project context.

**If memory files are uninitialized** (empty or bootstrap templates), the brief will show "ACTION REQUIRED" instructing you to run `/update-memory`.

**The /update-memory skill** analyzes git history and populates memory files with:
- Current state from recent commits
- Constraints learned from bugs/infrastructure
- Decisions with evidence (commit hashes)

See `\docs\MEMORY_SYSTEM_HOW_IT_WORKS.md` and `docs/PROJECT_MEMORY_SPEC` for details.

## Rules

- Follow the spec exactly
- Update this document after every meaningful commit
- Ask before adding dependencies
- When opening a repo, check if brief shows "ACTION REQUIRED" - if so, offer to run `/update-memory`
- [REPO_SPECIFIC_RULE]
- [REPO_SPECIFIC_RULE]