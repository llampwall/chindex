# tests/hooks/test_generator.py
import pytest
from pathlib import Path
from chinvex.hooks.generator import generate_post_commit_hook


def test_generate_creates_shell_wrapper(tmp_path: Path):
    """Should create .git/hooks/post-commit shell script"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    git_dir = repo_root / ".git"
    git_dir.mkdir()

    python_path = "C:\\Python312\\python.exe"
    context_name = "TestRepo"

    generate_post_commit_hook(repo_root, git_dir, python_path, context_name)

    hook_file = git_dir / "hooks" / "post-commit"
    assert hook_file.exists()

    content = hook_file.read_text()
    assert "#!/bin/sh" in content
    assert "pwsh" in content
    assert ".chinvex/post-commit.ps1" in content


def test_generate_creates_powershell_script(tmp_path: Path):
    """Should create .chinvex/post-commit.ps1"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    python_path = "C:\\Python312\\python.exe"
    context_name = "TestRepo"

    generate_post_commit_hook(repo_root, git_dir, python_path, context_name)

    ps_script = repo_root / ".chinvex" / "post-commit.ps1"
    assert ps_script.exists()

    content = ps_script.read_text()
    # Path gets normalized to forward slashes
    assert "C:/Python312/python.exe" in content
    assert "chinvex.cli" in content
    assert "ingest" in content
    assert f'"--context","{context_name}"' in content
    assert '"--paths"' in content


def test_generate_backs_up_existing_hook(tmp_path: Path):
    """Should backup existing post-commit hook"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True)

    # Create existing hook
    existing = hooks_dir / "post-commit"
    existing.write_text("#!/bin/sh\necho 'existing hook'\n")

    python_path = "python"
    context_name = "TestRepo"

    generate_post_commit_hook(repo_root, git_dir, python_path, context_name)

    # Original should be backed up
    backup_files = list(hooks_dir.glob("post-commit.bak*"))
    assert len(backup_files) == 1
    assert "existing hook" in backup_files[0].read_text()


def test_generated_hook_is_executable(tmp_path: Path):
    """Generated hook should have executable permissions (Unix only)"""
    import sys
    import os
    import stat

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    generate_post_commit_hook(repo_root, git_dir, "python", "TestRepo")

    hook_file = git_dir / "hooks" / "post-commit"
    assert hook_file.exists()

    # On Unix systems, verify execute permissions
    # On Windows, this is best-effort and may not set execute bit
    if sys.platform != 'win32':
        mode = os.stat(hook_file).st_mode
        assert mode & stat.S_IXUSR  # Owner execute
