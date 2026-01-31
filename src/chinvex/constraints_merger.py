# src/chinvex/constraints_merger.py
from __future__ import annotations

import datetime
import re


def extract_new_constraints(commits: list, specs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Extract potential new constraints from commits and specs.

    Uses heuristics to identify constraint-like statements.
    In production, this would use LLM to reason about what constitutes a constraint.

    Args:
        commits: List of GitCommit objects
        specs: List of spec dicts with 'path' and 'content'

    Returns:
        List of constraint dicts with 'section' and 'bullet' keys
    """
    constraints = []

    # Look for bounded inputs mentions in commits
    for commit in commits:
        if "max" in commit.message.lower() and ("limit" in commit.message.lower() or "commits" in commit.message.lower()):
            constraints.append({
                "section": "Infrastructure",
                "bullet": f"Bounded inputs: see spec for limits (from {commit.hash[:7]})"
            })

    # Look for limits in specs
    for spec in specs:
        content = spec["content"]
        if "max" in content.lower() and "kb" in content.lower():
            # Found size limit mention
            constraints.append({
                "section": "Infrastructure",
                "bullet": "Memory file update has bounded inputs (max commits, files, size)"
            })
            break

    return constraints


def move_to_superseded(content: str, obsolete_info: dict[str, str]) -> str:
    """Move a constraint from its section to Superseded.

    Args:
        content: Current CONSTRAINTS.md content
        obsolete_info: Dict with 'section', 'bullet', 'reason'

    Returns:
        Updated content with constraint moved to Superseded
    """
    section = obsolete_info["section"]
    bullet = obsolete_info["bullet"]
    reason = obsolete_info["reason"]
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Remove from original section
    # Find the bullet line and remove it
    bullet_pattern = re.escape(bullet)
    content = re.sub(rf"^- {bullet_pattern}\n", "", content, flags=re.MULTILINE)

    # Add to Superseded section
    superseded_marker = "## Superseded"
    if superseded_marker in content:
        # Find the section and append
        superseded_pos = content.find(superseded_marker)
        section_end = content.find("\n## ", superseded_pos + len(superseded_marker))
        if section_end == -1:
            section_end = len(content)

        # Insert before end of section
        insert_pos = section_end
        # Skip past "(None yet)" if present
        none_yet_pos = content.find("(None yet)", superseded_pos)
        if none_yet_pos > 0 and none_yet_pos < section_end:
            # Replace (None yet)
            content = content.replace("(None yet)\n", "")
            # Recalculate positions
            superseded_pos = content.find(superseded_marker)
            insert_pos = content.find("\n## ", superseded_pos)
            if insert_pos == -1:
                insert_pos = len(content)

        new_entry = f"- (Superseded {today}) {bullet} - {reason}\n"
        content = content[:insert_pos] + new_entry + content[insert_pos:]

    return content


def merge_constraints(current: str, new_constraints: list[dict[str, str]]) -> str:
    """Merge new constraints into CONSTRAINTS.md.

    Adds bullets to appropriate sections, avoids duplicates.

    Args:
        current: Current CONSTRAINTS.md content
        new_constraints: List of dicts with 'section' and 'bullet'

    Returns:
        Updated CONSTRAINTS.md content
    """
    content = current

    for constraint in new_constraints:
        section = constraint["section"]
        bullet = constraint["bullet"]

        # Check if bullet already exists (avoid duplicates)
        if bullet in content:
            continue

        # Find the section
        section_marker = f"## {section}"
        if section_marker not in content:
            # Section doesn't exist - skip or add section
            continue

        section_pos = content.find(section_marker)
        # Find next section or end of file
        next_section = content.find("\n## ", section_pos + len(section_marker))
        if next_section == -1:
            next_section = len(content)

        # Insert bullet before next section
        # Find last bullet in this section
        section_content = content[section_pos:next_section]
        last_bullet_match = None
        for match in re.finditer(r"^- .+$", section_content, re.MULTILINE):
            last_bullet_match = match

        if last_bullet_match:
            # Insert after last bullet
            insert_pos = section_pos + last_bullet_match.end() + 1
        else:
            # No bullets yet, insert after section header
            insert_pos = section_pos + len(section_marker) + 1

        content = content[:insert_pos] + f"- {bullet}\n" + content[insert_pos:]

    return content
