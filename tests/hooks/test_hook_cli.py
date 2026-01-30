"""Tests for hook CLI commands."""
import pytest
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_hook_install_in_git_repo(tmp_path: Path, monkeypatch):
    """hook install should create post-commit hook"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    result = runner.invoke(app, ["hook", "install", "--context", "TestRepo"])

    assert result.exit_code == 0
    assert (git_dir / "hooks" / "post-commit").exists()
    assert (repo_root / ".chinvex" / "post-commit.ps1").exists()


def test_hook_install_fails_outside_git_repo(tmp_path: Path, monkeypatch):
    """hook install should fail if not in git repo"""
    not_a_repo = tmp_path / "not_repo"
    not_a_repo.mkdir()

    monkeypatch.chdir(not_a_repo)

    result = runner.invoke(app, ["hook", "install", "--context", "TestRepo"])

    assert result.exit_code != 0
    assert "not a git repository" in result.stdout.lower()


def test_hook_install_infers_context_from_folder(tmp_path: Path, monkeypatch):
    """hook install without --context should infer from folder name"""
    repo_root = tmp_path / "my-project"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    result = runner.invoke(app, ["hook", "install"])

    assert result.exit_code == 0
    ps_script = repo_root / ".chinvex" / "post-commit.ps1"
    assert ps_script.exists()
    # Should use normalized folder name
    assert "my-project" in ps_script.read_text() or "MyProject" in ps_script.read_text()


def test_hook_uninstall_removes_hook(tmp_path: Path, monkeypatch):
    """hook uninstall should remove generated hook"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    # Install first
    runner.invoke(app, ["hook", "install", "--context", "TestRepo"])
    assert (git_dir / "hooks" / "post-commit").exists()

    # Uninstall
    result = runner.invoke(app, ["hook", "uninstall"])

    assert result.exit_code == 0
    assert not (git_dir / "hooks" / "post-commit").exists()
    assert not (repo_root / ".chinvex" / "post-commit.ps1").exists()


def test_hook_status_shows_installed_state(tmp_path: Path, monkeypatch):
    """hook status should show whether hook is installed"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    # Before install
    result = runner.invoke(app, ["hook", "status"])
    assert result.exit_code == 0
    assert "not installed" in result.stdout.lower()

    # After install
    runner.invoke(app, ["hook", "install", "--context", "TestRepo"])
    result = runner.invoke(app, ["hook", "status"])
    assert result.exit_code == 0
    assert "installed" in result.stdout.lower()
    assert "TestRepo" in result.stdout
