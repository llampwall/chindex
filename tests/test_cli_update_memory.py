# tests/test_cli_update_memory.py
import subprocess
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app


def test_update_memory_command_exists():
    """update-memory subcommand should be registered."""
    runner = CliRunner()
    result = runner.invoke(app, ["update-memory", "--help"])

    assert result.exit_code == 0
    assert "Update memory files" in result.output or "update-memory" in result.output


def test_update_memory_requires_context():
    """Should require --context argument."""
    runner = CliRunner()
    result = runner.invoke(app, ["update-memory"])

    assert result.exit_code != 0
    assert "context" in result.output.lower() or "required" in result.output.lower()


def test_update_memory_review_mode_shows_diff(tmp_path):
    """Review mode (default) should show diff without committing."""
    # Create test repo with context
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create chinvex config
    config_dir = tmp_path / ".chinvex" / "contexts" / "test_context"
    config_dir.mkdir(parents=True)
    context_file = config_dir / "context.json"
    context_file.write_text(f"""{{
        "context_name": "test_context",
        "includes": {{
            "repos": ["{repo.as_posix()}"]
        }}
    }}""")

    # Create initial commit and memory files
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result_run.stdout.strip()

    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"""# State

## Current Objective
Test

<!-- chinvex:last-commit:{first_hash} -->
""")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    # Create second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Run update-memory in review mode
    runner = CliRunner()
    result = runner.invoke(app, ["update-memory", "--context", "test_context"], env={"CHINVEX_CONTEXTS_ROOT": str(tmp_path / ".chinvex" / "contexts")})

    # Should show diff output
    assert result.exit_code == 0
    assert "diff" in result.output.lower() or "STATE.md" in result.output or "---" in result.output

    # Should NOT commit
    result_log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "update memory files" not in result_log.stdout.lower()


def test_update_memory_commit_mode_creates_commit(tmp_path):
    """Commit mode (--commit) should auto-commit changes."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    config_dir = tmp_path / ".chinvex" / "contexts" / "test_context"
    config_dir.mkdir(parents=True)
    (config_dir / "context.json").write_text(f"""{{
        "context_name": "test_context",
        "includes": {{"repos": ["{repo.as_posix()}"]}}
    }}""")

    # Initial commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result_run.stdout.strip()

    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"# State\n\n<!-- chinvex:last-commit:{first_hash} -->")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add memory files"], cwd=repo, check=True)

    # Second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Run update-memory with --commit
    runner = CliRunner()
    result = runner.invoke(app, ["update-memory", "--context", "test_context", "--commit"], env={"CHINVEX_CONTEXTS_ROOT": str(tmp_path / ".chinvex" / "contexts")})

    assert result.exit_code == 0

    # Should have created commit
    result_log = subprocess.run(["git", "log", "--oneline", "-n", "1"], cwd=repo, capture_output=True, text=True)
    assert "docs: update memory files" in result_log.stdout.lower() or "memory" in result_log.stdout.lower()


def test_update_memory_no_changes_message(tmp_path):
    """Should display message if no changes needed."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    config_dir = tmp_path / ".chinvex" / "contexts" / "test_context"
    config_dir.mkdir(parents=True)
    (config_dir / "context.json").write_text(f"""{{
        "context_name": "test_context",
        "includes": {{"repos": ["{repo.as_posix()}"]}}
    }}""")

    # Single commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    head_hash = result_run.stdout.strip()

    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"# State\n\n<!-- chinvex:last-commit:{head_hash} -->")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    # Run update-memory
    runner = CliRunner()
    result = runner.invoke(app, ["update-memory", "--context", "test_context"], env={"CHINVEX_CONTEXTS_ROOT": str(tmp_path / ".chinvex" / "contexts")})

    assert result.exit_code == 0
    assert "no new commits" in result.output.lower() or "up to date" in result.output.lower() or "0 commits" in result.output
