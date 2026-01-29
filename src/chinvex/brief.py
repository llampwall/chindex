from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from prometheus_client import Counter

BRIEF_GENERATED = Counter(
    "chinvex_brief_generated_total",
    "Total briefs generated",
    ["context"]
)


def generate_brief(
    context_name: str,
    state_md: Path | None,
    constraints_md: Path | None,
    decisions_md: Path | None,
    latest_digest: Path | None,
    watch_history_log: Path | None,
    output: Path
) -> None:
    """
    Generate session brief from memory files and recent activity.
    Missing files are silently skipped (graceful degradation).
    """
    BRIEF_GENERATED.labels(context=context_name).inc()

    lines = [
        f"# Session Brief: {context_name}",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]

    # STATE.md: full content
    if state_md and state_md.exists():
        state_content = _extract_state_sections(state_md)
        if state_content:
            lines.extend(state_content)
            lines.append("")

    # CONSTRAINTS.md: top section only (until first ##)
    if constraints_md and constraints_md.exists():
        constraints_content = _extract_constraints_top(constraints_md)
        if constraints_content:
            lines.append("## Constraints (highlights)")
            lines.extend(constraints_content)
            lines.append("")

    # DECISIONS.md: last 7 days
    if decisions_md and decisions_md.exists():
        recent_decisions = _extract_recent_decisions(decisions_md, days=7)
        if recent_decisions:
            lines.append("## Recent Decisions (7d)")
            lines.extend(recent_decisions)
            lines.append("")

    # Latest digest
    if latest_digest and latest_digest.exists():
        digest_summary = _extract_digest_summary(latest_digest)
        if digest_summary:
            lines.append("## Recent Activity")
            lines.extend(digest_summary)
            lines.append("")

    # Watch history: last 5 hits or 24h
    if watch_history_log and watch_history_log.exists():
        watch_summary = _extract_watch_summary(watch_history_log)
        if watch_summary:
            lines.append("## Recent Watch Hits")
            lines.extend(watch_summary)
            lines.append("")

    # Context files reference
    lines.append("## Context Files")
    if state_md:
        lines.append(f"- State: `{state_md}`")
    if latest_digest:
        lines.append(f"- Digest: `{latest_digest}`")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def _extract_state_sections(state_md: Path) -> list[str]:
    """Extract all sections from STATE.md."""
    content = state_md.read_text()
    lines = content.split("\n")

    result = []
    for line in lines:
        if line.startswith("# State"):
            continue  # Skip title
        result.append(line)

    return result


def _extract_constraints_top(constraints_md: Path) -> list[str]:
    """Extract content until first ## heading."""
    content = constraints_md.read_text()
    lines = content.split("\n")

    result = []
    seen_first_section = False

    for line in lines:
        if line.startswith("# Constraints"):
            continue
        if line.startswith("## "):
            if not seen_first_section:
                seen_first_section = True
                result.append(line)
                continue
            else:
                break  # Stop at second ## heading
        result.append(line)

    return result


def _extract_recent_decisions(decisions_md: Path, days: int) -> list[str]:
    """Extract decisions from last N days."""
    content = decisions_md.read_text()
    lines = content.split("\n")

    cutoff_date = datetime.now() - timedelta(days=days)
    result = []
    current_decision = []
    current_date = None

    for line in lines:
        # Match decision heading: ### YYYY-MM-DD — Title
        match = re.match(r"^### (\d{4}-\d{2}-\d{2}) — (.+)", line)
        if match:
            date_str, title = match.groups()
            decision_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Save previous decision if within window
            if current_decision and current_date and current_date >= cutoff_date:
                result.extend(current_decision)
                result.append("")

            # Start new decision
            current_date = decision_date
            current_decision = [line]
        elif current_decision:
            current_decision.append(line)

    # Save last decision
    if current_decision and current_date and current_date >= cutoff_date:
        result.extend(current_decision)

    return result


def _extract_digest_summary(digest_md: Path) -> list[str]:
    """Extract summary from latest digest."""
    content = digest_md.read_text()
    lines = content.split("\n")

    # Extract "Recent Changes" section
    result = []
    in_changes = False

    for line in lines:
        if line.startswith("## Recent Changes"):
            in_changes = True
            continue
        if in_changes:
            if line.startswith("##"):
                break
            result.append(line)

    return result


def _extract_watch_summary(watch_log: Path) -> list[str]:
    """Extract last 5 watch hits or 24h."""
    import json
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(hours=24)
    hits = []

    with watch_log.open("r") as f:
        for line in f:
            entry = json.loads(line.strip())
            ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
            if ts >= cutoff:
                hits.append(entry)

    # Take last 5
    hits = hits[-5:]

    result = []
    for hit in hits:
        result.append(f"- **\"{hit['query']}\"** ({len(hit['hits'])} hits)")

    return result
