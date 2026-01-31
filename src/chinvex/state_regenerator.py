# src/chinvex/state_regenerator.py
from __future__ import annotations

import datetime
import re


def update_coverage_anchor(content: str, new_hash: str) -> str:
    """Update coverage anchor in STATE.md content.

    Replaces existing anchor or appends if missing.

    Args:
        content: Current STATE.md content
        new_hash: New commit hash to use

    Returns:
        Updated content with new anchor
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Replace existing anchor
    anchor_pattern = r"<!-- chinvex:last-commit:[a-z0-9_]+ -->"
    if re.search(anchor_pattern, content):
        content = re.sub(anchor_pattern, f"<!-- chinvex:last-commit:{new_hash} -->", content)
    else:
        # Add anchor at end
        content = content.rstrip() + f"\n\n<!-- chinvex:last-commit:{new_hash} -->\n"

    # Update "Commits covered through" line
    commits_line_pattern = r"Commits covered through: [a-z0-9_]+"
    if re.search(commits_line_pattern, content):
        content = re.sub(commits_line_pattern, f"Commits covered through: {new_hash}", content)
    else:
        # Add before anchor
        anchor_pos = content.find("<!-- chinvex:last-commit:")
        if anchor_pos > 0:
            content = (
                content[:anchor_pos] +
                f"Commits covered through: {new_hash}\n\n" +
                content[anchor_pos:]
            )

    # Update timestamp
    timestamp_pattern = r"Last memory update: \d{4}-\d{2}-\d{2}"
    if re.search(timestamp_pattern, content):
        content = re.sub(timestamp_pattern, f"Last memory update: {today}", content)
    else:
        # Add before commits line
        commits_pos = content.find("Commits covered through:")
        if commits_pos > 0:
            content = (
                content[:commits_pos] +
                f"Last memory update: {today}\n" +
                content[commits_pos:]
            )

    return content


def regenerate_state_md(
    commits: list,
    specs: list[dict[str, str]],
    current_state: str,
    ending_commit_hash: str
) -> str:
    """Regenerate STATE.md based on commits and specs.

    This is a FULL regeneration - manual edits may be lost.
    Uses LLM-like logic to infer current objective from commits/specs.

    Args:
        commits: List of GitCommit objects
        specs: List of spec dicts with 'path' and 'content'
        current_state: Current STATE.md content (for reference)
        ending_commit_hash: Final commit hash for coverage anchor

    Returns:
        New STATE.md content
    """
    # Simple heuristic-based regeneration (placeholder for LLM call in real implementation)
    # In production, this would call an LLM to analyze commits+specs and generate content

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Attempt to infer objective from specs or commits
    objective = "Unknown (needs human)"
    active_work = []

    # Look for spec mentions in recent commits
    for commit in commits[:5]:  # Last 5 commits
        msg_lower = commit.message.lower()
        if "p5" in msg_lower or "memory" in msg_lower:
            objective = "P5 implementation - memory automation"
            active_work.append("Implementing memory file maintainer")
            break

    # Check specs for objective
    for spec in specs:
        if "P5" in spec["path"] and "memory" in spec["content"].lower():
            objective = "P5 implementation - memory automation"
            break

    # Build new STATE.md
    active_work_str = "\n".join([f"- {item}" for item in active_work]) if active_work else "- None"

    new_content = f"""# State

## Current Objective
{objective}

## Active Work
{active_work_str}

## Blockers
None

## Next Actions
- [ ] Run chinvex update-memory to refresh this file

## Out of Scope (for now)
- TBD

---
Last memory update: {today}
Commits covered through: {ending_commit_hash}

<!-- chinvex:last-commit:{ending_commit_hash} -->
"""

    return new_content
