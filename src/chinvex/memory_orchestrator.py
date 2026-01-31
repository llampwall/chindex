# src/chinvex/memory_orchestrator.py
from __future__ import annotations

import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from chinvex.git_analyzer import extract_coverage_anchor, get_commit_range
from chinvex.spec_reader import extract_spec_plan_files_from_commits, read_spec_files, BOUNDED_INPUTS
from chinvex.state_regenerator import regenerate_state_md
from chinvex.constraints_merger import merge_constraints, extract_new_constraints
from chinvex.decisions_appender import update_recent_rollup


@dataclass
class MemoryUpdateResult:
    """Result of a memory file update operation."""
    commits_processed: int
    files_analyzed: int
    files_changed: int
    bounded_inputs_triggered: bool
    ending_commit_hash: str


def get_memory_diff(filename: str, old_content: str, new_content: str) -> str:
    """Generate unified diff between old and new content.

    Args:
        filename: Name of file for diff header
        old_content: Original content
        new_content: Updated content

    Returns:
        Unified diff string
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    )

    return "".join(diff)


def update_memory_files(repo_root: Path) -> MemoryUpdateResult:
    """Update memory files based on git history.

    Main orchestration function that:
    1. Reads coverage anchor from STATE.md
    2. Gets commit range
    3. Reads specs/plans touched
    4. Updates STATE.md (full regen), CONSTRAINTS.md (merge), DECISIONS.md (append)
    5. Updates coverage anchor

    Args:
        repo_root: Repository root directory

    Returns:
        MemoryUpdateResult with metrics
    """
    memory_dir = repo_root / "docs" / "memory"
    state_file = memory_dir / "STATE.md"
    constraints_file = memory_dir / "CONSTRAINTS.md"
    decisions_file = memory_dir / "DECISIONS.md"

    # Read coverage anchor
    start_hash = None
    if state_file.exists():
        start_hash = extract_coverage_anchor(state_file)

    # Get commit range
    commits = get_commit_range(repo_root, start_hash=start_hash, max_commits=BOUNDED_INPUTS["max_commits"])

    if not commits:
        # No new commits
        return MemoryUpdateResult(
            commits_processed=0,
            files_analyzed=0,
            files_changed=0,
            bounded_inputs_triggered=False,
            ending_commit_hash=start_hash or "unknown"
        )

    # Get HEAD hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )
    ending_hash = result.stdout.strip()

    # Extract spec/plan files
    spec_files = extract_spec_plan_files_from_commits(repo_root, start_hash=start_hash)
    spec_content = read_spec_files(spec_files)

    bounded_inputs_triggered = spec_content.truncated_files or spec_content.truncated_size or len(commits) >= BOUNDED_INPUTS["max_commits"]

    # Read current memory files
    current_state = state_file.read_text() if state_file.exists() else ""
    current_constraints = constraints_file.read_text() if constraints_file.exists() else ""
    current_decisions = decisions_file.read_text() if decisions_file.exists() else ""

    # Update STATE.md (full regeneration)
    new_state = regenerate_state_md(
        commits=commits,
        specs=spec_content.specs,
        current_state=current_state,
        ending_commit_hash=ending_hash
    )

    # Update CONSTRAINTS.md (merge)
    new_constraints_list = extract_new_constraints(commits, spec_content.specs)
    new_constraints = merge_constraints(current_constraints, new_constraints_list)

    # Update DECISIONS.md (append + rollup)
    # For now, just update rollup (actual decision extraction is more complex)
    new_decisions = update_recent_rollup(current_decisions)

    # Write files
    files_changed = 0

    if new_state != current_state:
        state_file.write_text(new_state)
        files_changed += 1

    if new_constraints != current_constraints:
        constraints_file.write_text(new_constraints)
        files_changed += 1

    if new_decisions != current_decisions:
        decisions_file.write_text(new_decisions)
        files_changed += 1

    return MemoryUpdateResult(
        commits_processed=len(commits),
        files_analyzed=len(spec_content.specs),
        files_changed=files_changed,
        bounded_inputs_triggered=bounded_inputs_triggered,
        ending_commit_hash=ending_hash
    )
