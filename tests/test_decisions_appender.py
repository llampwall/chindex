# tests/test_decisions_appender.py
import datetime
from chinvex.decisions_appender import (
    append_decision,
    update_recent_rollup,
    ensure_month_section,
)
from chinvex.git_analyzer import GitCommit


def test_ensure_month_section_creates_if_missing():
    """Should create month section if it doesn't exist."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- Decision from January
"""

    # Request February section
    updated = ensure_month_section(current, "2026-02")

    assert "## 2026-02" in updated
    # Should be inserted after Recent section
    assert updated.find("## Recent") < updated.find("## 2026-02") < updated.find("## 2026-01")


def test_ensure_month_section_preserves_if_exists():
    """Should not duplicate existing month section."""
    current = """# Decisions

## Recent (last 30 days)
- Recent decision

## 2026-01
- Existing decision
"""

    updated = ensure_month_section(current, "2026-01")

    # Should only have one occurrence
    assert updated.count("## 2026-01") == 1
    assert "Existing decision" in updated


def test_append_decision_adds_to_month_section():
    """Should append decision to correct month section."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- Older decision
"""

    decision = {
        "date": "2026-01-31",
        "title": "Use OpenAI as default embedding provider",
        "rationale": "Better quality than Ollama for most use cases"
    }

    updated = append_decision(current, decision)

    # Should appear in 2026-01 section
    assert "Use OpenAI as default embedding provider" in updated
    assert "Better quality than Ollama" in updated
    assert "(2026-01-31)" in updated
    # Should be added to month section
    assert updated.find("## 2026-01") < updated.find("Use OpenAI")


def test_append_decision_creates_month_section_if_needed():
    """Should create month section if decision date is new month."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- January decision
"""

    decision = {
        "date": "2026-02-15",
        "title": "Switch to ChromaDB v2",
        "rationale": "Performance improvements"
    }

    updated = append_decision(current, decision)

    # Should create February section
    assert "## 2026-02" in updated
    assert "Switch to ChromaDB v2" in updated


def test_update_recent_rollup_includes_last_30_days():
    """Should update Recent section with decisions from last 30 days."""
    today = datetime.datetime.now()
    last_month = today - datetime.timedelta(days=25)
    two_months_ago = today - datetime.timedelta(days=60)

    current = f"""# Decisions

## Recent (last 30 days)
- TBD

## {today.strftime("%Y-%m")}
- (2026-01-31) Recent decision A
- ({two_months_ago.strftime("%Y-%m-%d")}) Old decision

## {last_month.strftime("%Y-%m")}
- ({last_month.strftime("%Y-%m-%d")}) Recent decision B
"""

    updated = update_recent_rollup(current)

    # Recent section should include only last 30 days
    recent_section_start = updated.find("## Recent")
    next_section = updated.find("\n## ", recent_section_start + 10)
    recent_content = updated[recent_section_start:next_section]

    assert "Recent decision A" in recent_content
    assert "Recent decision B" in recent_content
    assert "Old decision" not in recent_content


def test_update_recent_rollup_preserves_chronological_order():
    """Should list recent decisions in reverse chronological order (newest first)."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- (2026-01-31) Decision C
- (2026-01-20) Decision B
- (2026-01-10) Decision A
"""

    updated = update_recent_rollup(current)

    recent_section_start = updated.find("## Recent")
    next_section = updated.find("\n## ", recent_section_start + 10)
    recent_content = updated[recent_section_start:next_section]

    # Newest should come first
    pos_c = recent_content.find("Decision C")
    pos_b = recent_content.find("Decision B")
    pos_a = recent_content.find("Decision A")

    assert pos_c < pos_b < pos_a


def test_append_decision_format_matches_spec():
    """Decision entries should match format: (YYYY-MM-DD) Title - Rationale."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
(No decisions yet)
"""

    decision = {
        "date": "2026-01-31",
        "title": "Max 50 commits per memory update",
        "rationale": "Bounded inputs guardrail to prevent timeout"
    }

    updated = append_decision(current, decision)

    # Check format
    assert "- (2026-01-31) Max 50 commits per memory update - Bounded inputs guardrail to prevent timeout" in updated


def test_update_recent_rollup_handles_empty_months():
    """Should skip empty month sections when building Recent rollup."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- (2026-01-31) Decision A

## 2025-12
(No decisions recorded yet)
"""

    updated = update_recent_rollup(current)

    # Should only include Decision A
    recent_section_start = updated.find("## Recent")
    next_section = updated.find("\n## ", recent_section_start + 10)
    recent_content = updated[recent_section_start:next_section]

    assert "Decision A" in recent_content
    # Count bullet points (lines starting with "- (")
    bullet_count = recent_content.count("\n- (")
    assert bullet_count == 1  # Only one decision
