from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    "out",
    ".venv",
    "venv",
    ".codex",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    ".eggs",
    "chroma",
    ".worktrees",
    ".claude",
    ".pnpm-store",
    ".pnpm",
    ".vscode",
    ".ruff_cache",
    ".cursor",
    ".qodo",
    "logs",
    "log",
}

ALLOWED_EXTS = {".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_path(path: Path) -> str:
    return str(path.resolve())


def iso_from_mtime(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def read_text_utf8(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def should_exclude(path: Path, root: Path, excludes: list[str]) -> bool:
    """Check if path matches any exclude pattern relative to root."""
    rel_path = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in excludes)


def walk_files(root: Path, excludes: list[str] | None = None) -> Iterable[Path]:
    """Walk files in root, skipping SKIP_DIRS and paths matching exclude patterns."""
    excludes = excludes or []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in ALLOWED_EXTS and not should_exclude(path, root, excludes):
                yield path


def dump_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def in_venv() -> bool:
    return os.environ.get("VIRTUAL_ENV") is not None or getattr(os, "base_prefix", "") != getattr(os, "prefix", "")


def dataclass_to_json(obj: object) -> str:
    return dump_json(asdict(obj))


def normalize_path_for_dedup(path: str | Path) -> str:
    """
    Normalize path for deduplication:
    - Convert to absolute
    - Use forward slashes
    - Lowercase on Windows (case-insensitive)
    """
    abs_path = Path(path).resolve()
    normalized = abs_path.as_posix()

    if platform.system() == "Windows":
        normalized = normalized.lower()

    return normalized


def backup_context_json(context_file: Path) -> None:
    """
    Back up context.json before write operations.

    Creates backup at P:/ai_memory/backups/<name>/context-{timestamp}.json
    Keeps max 30 backups per context, pruning oldest.

    Args:
        context_file: Path to the context.json file to back up
    """
    # Only backup if file exists
    if not context_file.exists():
        return

    # Extract context name from path (parent directory name)
    context_name = context_file.parent.name

    # Get backups root
    backups_root = Path(os.getenv("CHINVEX_BACKUPS_ROOT", "P:/ai_memory/backups"))
    backup_dir = backups_root / context_name

    # Create backup directory if missing
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp in YYYYMMDD-HHMMSS format with milliseconds to prevent collisions
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:19]  # YYYYMMDD-HHMMSS-mmm (milliseconds)
    backup_file = backup_dir / f"context-{timestamp}.json"

    # Copy file with metadata preservation
    shutil.copy2(context_file, backup_file)

    # Prune old backups - keep max 30
    backup_files = sorted(backup_dir.glob("context-*.json"))
    if len(backup_files) > 30:
        # Remove oldest files
        for old_backup in backup_files[:-30]:
            old_backup.unlink()
