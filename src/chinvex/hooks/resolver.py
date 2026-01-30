"""Python path resolution for git hooks."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_python_path(repo_root: Path) -> str | None:
    """
    Resolve Python executable path with chinvex installed.

    Order of preference:
    1. CHINVEX_PYTHON env var (explicit override)
    2. {repo}/.venv/Scripts/python.exe (repo-local venv)
    3. py -3 (Windows Python launcher)
    4. python (PATH fallback)

    Args:
        repo_root: Repository root directory

    Returns:
        Path to Python executable, or None if not found
    """
    candidates = []

    # 1. Explicit override
    env_python = os.getenv("CHINVEX_PYTHON")
    if env_python:
        candidates.append(env_python)

    # 2. Repo-local venv
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        candidates.append(str(venv_python))

    # Also check Unix-style venv
    venv_python_unix = repo_root / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        candidates.append(str(venv_python_unix))

    # 3. Windows Python launcher
    if shutil.which("py"):
        candidates.append("py -3")

    # 4. PATH fallback
    if shutil.which("python"):
        candidates.append("python")

    # Test each candidate
    for candidate in candidates:
        if _validate_python(candidate):
            log.info(f"Resolved Python: {candidate}")
            return candidate

    log.error("No valid Python with chinvex found")
    return None


def _validate_python(python_path: str) -> bool:
    """
    Validate that Python works and has chinvex installed.

    Args:
        python_path: Python executable path or command

    Returns:
        True if valid and has chinvex
    """
    # Handle "py -3" style commands
    if " " in python_path:
        cmd_parts = python_path.split()
    else:
        cmd_parts = [python_path]

    try:
        # Check Python version
        result = subprocess.run(
            cmd_parts + ["--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False

        # Check chinvex module exists
        result = subprocess.run(
            cmd_parts + ["-m", "chinvex.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
