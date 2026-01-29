import pytest
from pathlib import Path
from datetime import datetime, timedelta


def test_generate_brief_minimal(tmp_path):
    """Test brief generation with minimal inputs."""
    from chinvex.brief import generate_brief

    output = tmp_path / "SESSION_BRIEF.md"

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    assert output.exists()
    content = output.read_text()
    assert "# Session Brief: TestContext" in content
    assert "Generated:" in content


def test_generate_brief_with_state(tmp_path):
    """Test brief includes STATE.md content."""
    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
P4 implementation

## Active Work
- Digest generation
- Brief generation

## Blockers
None

## Next Actions
- [ ] Complete P4.4
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=state_md,
        constraints_md=None,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()
    assert "Current Objective" in content
    assert "P4 implementation" in content
    assert "Active Work" in content


def test_generate_brief_with_constraints(tmp_path):
    """Test brief includes CONSTRAINTS.md top section."""
    constraints_md = tmp_path / "CONSTRAINTS.md"
    constraints_md.write_text("""# Constraints

## Infrastructure
- ChromaDB batch limit: 5000
- Embedding dims locked

## Rules
- Schema stays v2
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=constraints_md,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()
    assert "Constraints" in content
    assert "ChromaDB batch limit" in content


def test_generate_brief_with_recent_decisions(tmp_path):
    """Test brief includes recent decisions (last 7 days)."""
    recent_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    decisions_md = tmp_path / "DECISIONS.md"
    decisions_md.write_text(f"""# Decisions

### {recent_date} — Recent decision

- **Why:** Testing
- **Impact:** Should appear
- **Evidence:** commit abc123

### {old_date} — Old decision

- **Why:** Testing
- **Impact:** Should NOT appear
- **Evidence:** commit def456
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=decisions_md,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()
    assert "Recent decision" in content
    assert "Old decision" not in content
