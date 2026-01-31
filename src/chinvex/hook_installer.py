# src/chinvex/hook_installer.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def is_git_repo(directory: Path) -> bool:
    """Check if directory is a git repository.

    Args:
        directory: Path to check

    Returns:
        True if directory is a git repo
    """
    git_dir = directory / ".git"
    if git_dir.exists():
        return True

    # Check via git command
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=directory,
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def merge_settings_json(base: dict, overlay: dict) -> dict:
    """Deep merge two settings dictionaries.

    Args:
        base: Base settings
        overlay: Settings to merge in

    Returns:
        Merged settings dict
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Deep merge nested dicts
            result[key] = merge_settings_json(result[key], value)
        else:
            result[key] = value

    return result


def install_startup_hook(repo_root: Path, context_name: str) -> bool:
    """Install Claude Code startup hook in .claude/settings.json.

    Creates or updates settings.json to include:
    {
        "hooks": {
            "startup": ["chinvex brief --context <context_name>"]
        }
    }

    Args:
        repo_root: Repository root directory
        context_name: Chinvex context name

    Returns:
        True if hook installed successfully, False otherwise
    """
    # Check if git repo
    if not is_git_repo(repo_root):
        print(f"Warning: {repo_root} is not a git repository. Skipping hook installation.")
        return False

    claude_dir = repo_root / ".claude"
    claude_dir.mkdir(exist_ok=True)

    settings_file = claude_dir / "settings.json"
    hook_command = f"chinvex brief --context {context_name}"

    # Read existing settings
    if settings_file.exists():
        try:
            existing = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            print(f"Warning: Malformed {settings_file}. Skipping hook installation.")
            return False
    else:
        existing = {}

    # Prepare hook update
    if "hooks" not in existing:
        existing["hooks"] = {}

    if "startup" not in existing["hooks"]:
        existing["hooks"]["startup"] = []
    elif isinstance(existing["hooks"]["startup"], str):
        # Convert string to array
        existing["hooks"]["startup"] = [existing["hooks"]["startup"]]

    # Add hook if not already present
    if hook_command not in existing["hooks"]["startup"]:
        existing["hooks"]["startup"].append(hook_command)

    # Write back
    settings_file.write_text(json.dumps(existing, indent=2) + "\n")
    return True
