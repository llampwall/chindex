# src/chinvex/decisions_appender.py
from __future__ import annotations

import datetime
import re


def ensure_month_section(content: str, month_key: str) -> str:
    """Ensure month section exists in DECISIONS.md.

    Month sections are inserted in reverse chronological order after Recent section.

    Args:
        content: Current DECISIONS.md content
        month_key: Month in YYYY-MM format

    Returns:
        Updated content with month section present
    """
    month_marker = f"## {month_key}"

    if month_marker in content:
        return content  # Already exists

    # Find Recent section
    recent_marker = "## Recent (last 30 days)"
    if recent_marker not in content:
        # Malformed file
        return content

    recent_pos = content.find(recent_marker)
    # Find next section after Recent
    next_section_pos = content.find("\n## ", recent_pos + len(recent_marker))

    if next_section_pos == -1:
        # No sections after Recent - append at end
        insert_pos = len(content)
    else:
        insert_pos = next_section_pos + 1

    # Insert new month section
    new_section = f"## {month_key}\n(No decisions recorded yet)\n\n"
    content = content[:insert_pos] + new_section + content[insert_pos:]

    return content


def append_decision(content: str, decision: dict[str, str]) -> str:
    """Append a decision to DECISIONS.md.

    Adds decision to the appropriate month section (creating if needed).

    Args:
        content: Current DECISIONS.md content
        decision: Dict with 'date' (YYYY-MM-DD), 'title', 'rationale'

    Returns:
        Updated content with decision added
    """
    date_str = decision["date"]
    title = decision["title"]
    rationale = decision["rationale"]

    # Extract month from date
    month_key = date_str[:7]  # YYYY-MM

    # Ensure month section exists
    content = ensure_month_section(content, month_key)

    # Format decision entry
    entry = f"- ({date_str}) {title} - {rationale}\n"

    # Find month section
    month_marker = f"## {month_key}"
    month_pos = content.find(month_marker)

    if month_pos == -1:
        return content  # Shouldn't happen after ensure_month_section

    # Find next section or end of file
    next_section = content.find("\n## ", month_pos + len(month_marker))
    if next_section == -1:
        next_section = len(content)

    # Remove "(No decisions recorded yet)" placeholder if present
    section_content = content[month_pos:next_section]
    if "(No decisions recorded yet)" in section_content:
        content = content.replace("(No decisions recorded yet)\n", "")
        # Recalculate positions
        month_pos = content.find(month_marker)
        next_section = content.find("\n## ", month_pos + len(month_marker))
        if next_section == -1:
            next_section = len(content)

    # Insert decision at end of month section (before next section)
    insert_pos = next_section
    content = content[:insert_pos] + entry + content[insert_pos:]

    return content


def update_recent_rollup(content: str) -> str:
    """Update Recent section with decisions from last 30 days.

    Scans all month sections, extracts decisions from last 30 days,
    and updates the Recent rollup section.

    Args:
        content: Current DECISIONS.md content

    Returns:
        Updated content with Recent section refreshed
    """
    today = datetime.datetime.now()
    cutoff = today - datetime.timedelta(days=30)

    # Find all decision entries across all month sections
    # Match pattern: - (YYYY-MM-DD) Title - Rationale
    decision_pattern = re.compile(r"^- \((\d{4}-\d{2}-\d{2})\) (.+)$", re.MULTILINE)

    recent_decisions = []
    for match in decision_pattern.finditer(content):
        date_str = match.group(1)
        full_entry = match.group(2)

        # Parse date
        try:
            decision_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Check if within last 30 days
        if decision_date >= cutoff:
            recent_decisions.append({
                "date": date_str,
                "entry": f"- ({date_str}) {full_entry}",
                "datetime": decision_date
            })

    # Sort by date (newest first)
    recent_decisions.sort(key=lambda x: x["datetime"], reverse=True)

    # Build Recent section content
    if recent_decisions:
        recent_content = "\n".join([d["entry"] for d in recent_decisions]) + "\n"
    else:
        recent_content = "- (None in last 30 days)\n"

    # Replace Recent section
    recent_marker = "## Recent (last 30 days)"
    recent_pos = content.find(recent_marker)

    if recent_pos == -1:
        return content  # Malformed

    # Find next section
    next_section = content.find("\n## ", recent_pos + len(recent_marker))
    if next_section == -1:
        next_section = len(content)

    # Replace content between marker and next section
    before = content[:recent_pos + len(recent_marker)]
    after = content[next_section:]

    content = before + "\n" + recent_content + "\n" + after

    return content
