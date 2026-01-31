# tests/test_state_regenerator.py
from pathlib import Path
from chinvex.state_regenerator import (
    regenerate_state_md,
    update_coverage_anchor,
)
from chinvex.git_analyzer import GitCommit


def test_regenerate_state_md_creates_new_content():
    """Should generate STATE.md based on commits and specs."""
    commits = [
        GitCommit(
            hash="abc123",
            date="2026-01-31",
            author="Test",
            message="feat(P5): implement memory maintainer\n\nAdds update-memory command"
        ),
        GitCommit(
            hash="def456",
            date="2026-01-30",
            author="Test",
            message="docs: update P5 spec"
        )
    ]

    specs = [
        {"path": "specs/P5_SPEC.md", "content": "# P5 Spec\n\n## Goal\nImplement memory automation"}
    ]

    current_state = """# State

## Current Objective
Old objective

## Active Work
- Old work

## Blockers
None

## Next Actions
- [ ] Old action

## Out of Scope (for now)
- TBD
"""

    new_state = regenerate_state_md(
        commits=commits,
        specs=specs,
        current_state=current_state,
        ending_commit_hash="abc123"
    )

    # Should include coverage anchor
    assert "<!-- chinvex:last-commit:abc123 -->" in new_state
    # Should have structure
    assert "# State" in new_state
    assert "## Current Objective" in new_state
    assert "## Active Work" in new_state
    # Should reference recent work
    assert "memory" in new_state.lower() or "P5" in new_state


def test_update_coverage_anchor_replaces_existing():
    """Should replace existing anchor with new commit hash."""
    old_content = """# State

## Current Objective
Test

---
Last memory update: 2026-01-30
Commits covered through: old_hash

<!-- chinvex:last-commit:old_hash -->
"""

    new_content = update_coverage_anchor(old_content, new_hash="new_hash")

    assert "<!-- chinvex:last-commit:new_hash -->" in new_content
    assert "<!-- chinvex:last-commit:old_hash -->" not in new_content
    assert "Commits covered through: new_hash" in new_content


def test_update_coverage_anchor_adds_if_missing():
    """Should add anchor if not present."""
    old_content = """# State

## Current Objective
Test

## Active Work
- Work item
"""

    new_content = update_coverage_anchor(old_content, new_hash="abc123")

    assert "<!-- chinvex:last-commit:abc123 -->" in new_content
    assert "Commits covered through: abc123" in new_content


def test_regenerate_state_md_handles_no_objective_found():
    """If no clear objective can be determined, should use placeholder."""
    commits = [
        GitCommit(hash="abc", date="2026-01-31", author="Test", message="misc: update readme")
    ]

    specs = []
    current_state = "# State\n\n## Current Objective\nOld"

    new_state = regenerate_state_md(
        commits=commits,
        specs=specs,
        current_state=current_state,
        ending_commit_hash="abc"
    )

    # Should have fallback objective
    assert "Unknown (needs human)" in new_state or "TBD" in new_state


def test_regenerate_state_md_includes_timestamp():
    """Should update timestamp in footer."""
    import datetime

    commits = []
    specs = []
    current_state = "# State\n\n## Current Objective\nTest"

    new_state = regenerate_state_md(
        commits=commits,
        specs=specs,
        current_state=current_state,
        ending_commit_hash="abc"
    )

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    assert f"Last memory update: {today}" in new_state
