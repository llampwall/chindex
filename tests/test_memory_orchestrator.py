# tests/test_memory_orchestrator.py
import subprocess
from pathlib import Path
from chinvex.memory_orchestrator import (
    update_memory_files,
    get_memory_diff,
    MemoryUpdateResult,
)


def test_update_memory_files_reads_coverage_anchor(tmp_path):
    """Should read last commit hash from STATE.md coverage anchor."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create initial commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result.stdout.strip()

    # Create memory dir with STATE.md
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    state_file = memory_dir / "STATE.md"
    state_file.write_text(f"""# State

## Current Objective
Test

---
Last memory update: 2026-01-30
Commits covered through: {first_hash}

<!-- chinvex:last-commit:{first_hash} -->
""")

    # Create second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Run update
    result = update_memory_files(repo)

    # Should have processed one commit
    assert result.commits_processed == 1
    assert result.ending_commit_hash != first_hash


def test_update_memory_files_early_exit_if_no_new_commits(tmp_path):
    """Should return early if no new commits since last anchor."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    head_hash = result.stdout.strip()

    # Create memory files with coverage anchor at HEAD
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"""# State

<!-- chinvex:last-commit:{head_hash} -->
""")

    result = update_memory_files(repo)

    # Should detect no new commits
    assert result.commits_processed == 0
    assert result.files_changed == 0


def test_update_memory_files_updates_all_three_files(tmp_path):
    """Should update STATE.md, CONSTRAINTS.md, and DECISIONS.md."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # First commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result.stdout.strip()

    # Create memory files
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"""# State

## Current Objective
Old

<!-- chinvex:last-commit:{first_hash} -->
""")
    (memory_dir / "CONSTRAINTS.md").write_text("""# Constraints

## Infrastructure
- Old constraint

## Superseded
(None yet)
""")
    (memory_dir / "DECISIONS.md").write_text("""# Decisions

## Recent (last 30 days)
- TBD
""")

    # Second commit with spec change
    specs_dir = repo / "specs"
    specs_dir.mkdir()
    (specs_dir / "P1.md").write_text("# P1 Spec\n\nMax file size: 500 KB")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feat(P1): add spec"], cwd=repo, check=True)

    result = update_memory_files(repo)

    # Should update all files
    assert result.files_changed == 3
    assert (memory_dir / "STATE.md").exists()
    assert (memory_dir / "CONSTRAINTS.md").exists()
    assert (memory_dir / "DECISIONS.md").exists()


def test_get_memory_diff_returns_unified_diff(tmp_path):
    """Should return git-style unified diff of changes."""
    old_state = """# State

## Current Objective
Old objective

## Active Work
- Old work
"""

    new_state = """# State

## Current Objective
New objective

## Active Work
- New work
"""

    diff = get_memory_diff("STATE.md", old_state, new_state)

    assert "STATE.md" in diff
    assert "-Old objective" in diff or "- Old objective" in diff
    assert "+New objective" in diff or "+ New objective" in diff


def test_update_memory_files_respects_bounded_inputs(tmp_path):
    """Should handle bounded inputs limits gracefully."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create 60 commits (exceeds max_commits limit of 50)
    for i in range(60):
        (repo / f"file{i}.txt").write_text(f"commit {i}")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=repo, check=True, capture_output=True)

    # Create memory files (no anchor - should process from beginning)
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text("# State\n\n## Current Objective\nTest")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    result = update_memory_files(repo)

    # Should limit to max_commits
    assert result.commits_processed <= 50
    assert result.bounded_inputs_triggered is True


def test_memory_update_result_tracks_metrics():
    """MemoryUpdateResult should track all relevant metrics."""
    result = MemoryUpdateResult(
        commits_processed=5,
        files_analyzed=3,
        files_changed=2,
        bounded_inputs_triggered=False,
        ending_commit_hash="abc123"
    )

    assert result.commits_processed == 5
    assert result.files_analyzed == 3
    assert result.files_changed == 2
    assert result.bounded_inputs_triggered is False
    assert result.ending_commit_hash == "abc123"
