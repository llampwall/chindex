# tests/test_hook_installer.py
import json
import subprocess
from pathlib import Path
from chinvex.hook_installer import (
    install_startup_hook,
    merge_settings_json,
    is_git_repo,
)


def test_is_git_repo_detects_git_directory(tmp_path):
    """Should detect if directory is a git repo."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    assert is_git_repo(repo) is True


def test_is_git_repo_returns_false_for_non_git(tmp_path):
    """Should return False for non-git directory."""
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()

    assert is_git_repo(non_repo) is False


def test_install_startup_hook_creates_settings_json(tmp_path):
    """Should create .claude/settings.json if it doesn't exist."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    install_startup_hook(repo, context_name="test_context")

    settings_file = repo / ".claude" / "settings.json"
    assert settings_file.exists()

    settings = json.loads(settings_file.read_text())
    assert "hooks" in settings
    assert "startup" in settings["hooks"]
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]


def test_install_startup_hook_merges_with_existing_settings(tmp_path):
    """Should merge with existing settings.json without clobbering."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    # Create existing settings
    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "theme": "dark",
        "other_config": "value"
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should preserve existing config
    assert settings["theme"] == "dark"
    assert settings["other_config"] == "value"

    # Should add hook
    assert "hooks" in settings
    assert "startup" in settings["hooks"]


def test_install_startup_hook_converts_string_to_array(tmp_path):
    """Should convert existing string startup hook to array."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "startup": "existing-command"
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should be converted to array
    assert isinstance(settings["hooks"]["startup"], list)
    assert "existing-command" in settings["hooks"]["startup"]
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]


def test_install_startup_hook_appends_to_existing_array(tmp_path):
    """Should append to existing startup array without duplicating."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "startup": ["other-command"]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    assert "other-command" in settings["hooks"]["startup"]
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]
    assert len(settings["hooks"]["startup"]) == 2


def test_install_startup_hook_avoids_duplicates(tmp_path):
    """Should not add duplicate hook if already present."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "startup": ["chinvex brief --context test_context"]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should not duplicate
    assert settings["hooks"]["startup"].count("chinvex brief --context test_context") == 1


def test_install_startup_hook_skips_non_git_with_warning(tmp_path, caplog):
    """Should skip non-git directory and log warning."""
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()

    result = install_startup_hook(non_repo, context_name="test_context")

    assert result is False
    assert not (non_repo / ".claude" / "settings.json").exists()


def test_install_startup_hook_handles_malformed_json(tmp_path, caplog):
    """Should skip malformed settings.json and log warning."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text("{ invalid json }")

    result = install_startup_hook(repo, context_name="test_context")

    # Should fail gracefully
    assert result is False


def test_merge_settings_json_deep_merges():
    """Should deep merge settings without clobbering nested objects."""
    base = {
        "theme": "dark",
        "hooks": {
            "pre-commit": ["lint"]
        }
    }

    overlay = {
        "hooks": {
            "startup": ["brief"]
        },
        "new_field": "value"
    }

    merged = merge_settings_json(base, overlay)

    # Should preserve both hook types
    assert merged["hooks"]["pre-commit"] == ["lint"]
    assert merged["hooks"]["startup"] == ["brief"]
    assert merged["theme"] == "dark"
    assert merged["new_field"] == "value"
