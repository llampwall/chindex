# tests/test_constraints_merger.py
from chinvex.constraints_merger import (
    merge_constraints,
    extract_new_constraints,
    move_to_superseded,
)
from chinvex.git_analyzer import GitCommit


def test_merge_constraints_adds_new_bullets():
    """Should add new constraint bullets without removing existing ones."""
    current = """# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors
- Gateway port: 7778

## Rules
- Schema stays v2

## Key Facts
- Token env var: CHINVEX_API_TOKEN

## Hazards
- None

## Superseded
(None yet)
"""

    new_constraints = [
        {"section": "Infrastructure", "bullet": "Embedding dims locked per index"},
        {"section": "Rules", "bullet": "No migrations without rebuild"}
    ]

    updated = merge_constraints(current, new_constraints)

    # Should preserve existing
    assert "ChromaDB batch limit: 5000 vectors" in updated
    assert "Gateway port: 7778" in updated
    # Should add new
    assert "Embedding dims locked per index" in updated
    assert "No migrations without rebuild" in updated


def test_extract_new_constraints_from_commits():
    """Should identify potential new constraints from commit messages."""
    commits = [
        GitCommit(
            hash="abc",
            date="2026-01-31",
            author="Test",
            message="fix: respect max 50 commits limit\n\nBounded inputs guardrail"
        ),
        GitCommit(
            hash="def",
            date="2026-01-30",
            author="Test",
            message="feat: add new search endpoint"
        )
    ]

    specs = [
        {"path": "specs/P5.md", "content": "Max 50 commits per run\nMax 100KB total spec content"}
    ]

    constraints = extract_new_constraints(commits, specs)

    # Should extract bounded inputs
    assert len(constraints) > 0
    # Should have section and bullet
    assert all("section" in c and "bullet" in c for c in constraints)


def test_move_to_superseded():
    """Should move obsolete constraint to Superseded section with date and reason."""
    import datetime

    current = """# Constraints

## Infrastructure
- Old batch limit: 1000 vectors
- Gateway port: 7778

## Rules
- Schema stays v2

## Superseded
(None yet)
"""

    obsolete_info = {
        "section": "Infrastructure",
        "bullet": "Old batch limit: 1000 vectors",
        "reason": "Increased to 5000 in P3"
    }

    updated = move_to_superseded(current, obsolete_info)

    # Should remove from original section
    assert "## Infrastructure\n- Old batch limit: 1000 vectors" not in updated
    # Should appear in Superseded with date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    assert "## Superseded" in updated
    assert f"(Superseded {today})" in updated
    assert "Old batch limit: 1000 vectors" in updated
    assert "Increased to 5000 in P3" in updated


def test_merge_constraints_preserves_structure():
    """Should maintain all core sections even if empty."""
    current = """# Constraints

## Infrastructure
- TBD

## Rules
- TBD

## Key Facts
- TBD

## Hazards
- TBD

## Superseded
(None yet)
"""

    updated = merge_constraints(current, [])

    # All sections should remain
    assert "## Infrastructure" in updated
    assert "## Rules" in updated
    assert "## Key Facts" in updated
    assert "## Hazards" in updated
    assert "## Superseded" in updated


def test_merge_constraints_avoids_duplicates():
    """Should not add constraint if bullet already exists."""
    current = """# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors

## Rules
- Schema stays v2

## Superseded
(None yet)
"""

    new_constraints = [
        {"section": "Infrastructure", "bullet": "ChromaDB batch limit: 5000 vectors"},  # duplicate
        {"section": "Infrastructure", "bullet": "Gateway port: 7778"}  # new
    ]

    updated = merge_constraints(current, new_constraints)

    # Should only have one occurrence of batch limit
    assert updated.count("ChromaDB batch limit: 5000 vectors") == 1
    # Should add the new one
    assert "Gateway port: 7778" in updated
