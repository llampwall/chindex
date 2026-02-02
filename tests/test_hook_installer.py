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
    assert "SessionStart" in settings["hooks"]

    # Check that the hook command exists in the correct format
    hook_found = False
    for hook_group in settings["hooks"]["SessionStart"]:
        if "hooks" in hook_group:
            for hook in hook_group["hooks"]:
                if hook.get("type") == "command" and hook.get("command") == "chinvex brief --context test_context":
                    hook_found = True
                    break
    assert hook_found


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
    assert "SessionStart" in settings["hooks"]


def test_install_startup_hook_appends_to_existing_session_start(tmp_path):
    """Should append to existing SessionStart hooks without clobbering."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "existing-command"
                        }
                    ]
                }
            ]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should have both hooks
    assert len(settings["hooks"]["SessionStart"]) == 2

    # Verify both commands are present
    commands = []
    for hook_group in settings["hooks"]["SessionStart"]:
        if "hooks" in hook_group:
            for hook in hook_group["hooks"]:
                if hook.get("type") == "command":
                    commands.append(hook.get("command"))

    assert "existing-command" in commands
    assert "chinvex brief --context test_context" in commands


def test_install_startup_hook_preserves_other_hooks(tmp_path):
    """Should preserve other hook types when adding SessionStart."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "other-hook-command"
                        }
                    ]
                }
            ]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should preserve existing hook type
    assert "UserPromptSubmit" in settings["hooks"]

    # Should add SessionStart
    assert "SessionStart" in settings["hooks"]


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
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "chinvex brief --context test_context"
                        }
                    ]
                }
            ]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should not duplicate - should still have only 1 SessionStart hook group
    assert len(settings["hooks"]["SessionStart"]) == 1

    # Verify the command is still there
    hook_found = False
    for hook_group in settings["hooks"]["SessionStart"]:
        if "hooks" in hook_group:
            for hook in hook_group["hooks"]:
                if hook.get("type") == "command" and hook.get("command") == "chinvex brief --context test_context":
                    hook_found = True
                    break
    assert hook_found


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
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "lint"}]}]
        }
    }

    overlay = {
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": "brief"}]}]
        },
        "new_field": "value"
    }

    merged = merge_settings_json(base, overlay)

    # Should preserve both hook types
    assert merged["hooks"]["UserPromptSubmit"] == [{"hooks": [{"type": "command", "command": "lint"}]}]
    assert merged["hooks"]["SessionStart"] == [{"hooks": [{"type": "command", "command": "brief"}]}]
    assert merged["theme"] == "dark"
    assert merged["new_field"] == "value"
