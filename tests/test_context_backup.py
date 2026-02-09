"""Test context.json backup functionality."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from chinvex.util import backup_context_json


def test_backup_context_json_creates_backup(tmp_path: Path) -> None:
    """Test that backup_context_json creates a backup file."""
    # Setup
    context_dir = tmp_path / "contexts" / "test-context"
    context_dir.mkdir(parents=True)
    context_file = context_dir / "context.json"
    context_file.write_text(json.dumps({"name": "test", "version": 1}), encoding="utf-8")

    # Set backup root to tmp_path
    import os
    os.environ["CHINVEX_BACKUPS_ROOT"] = str(tmp_path / "backups")

    # Execute
    backup_context_json(context_file)

    # Verify
    backup_dir = tmp_path / "backups" / "test-context"
    assert backup_dir.exists()
    backup_files = list(backup_dir.glob("context-*.json"))
    assert len(backup_files) == 1

    # Verify backup content matches original
    backup_content = json.loads(backup_files[0].read_text(encoding="utf-8"))
    assert backup_content == {"name": "test", "version": 1}


def test_backup_context_json_skips_if_not_exists(tmp_path: Path) -> None:
    """Test that backup_context_json does nothing if file doesn't exist."""
    # Setup
    context_file = tmp_path / "contexts" / "test" / "context.json"

    # Set backup root to tmp_path
    import os
    os.environ["CHINVEX_BACKUPS_ROOT"] = str(tmp_path / "backups")

    # Execute - should not raise
    backup_context_json(context_file)

    # Verify no backups created
    backup_dir = tmp_path / "backups"
    assert not backup_dir.exists()


def test_backup_context_json_prunes_old_backups(tmp_path: Path) -> None:
    """Test that backup_context_json keeps max 30 backups."""
    # Setup
    context_dir = tmp_path / "contexts" / "test-context"
    context_dir.mkdir(parents=True)
    context_file = context_dir / "context.json"
    context_file.write_text(json.dumps({"name": "test"}), encoding="utf-8")

    # Set backup root to tmp_path
    import os
    os.environ["CHINVEX_BACKUPS_ROOT"] = str(tmp_path / "backups")

    # Create 32 backups (should prune to 30)
    for i in range(32):
        backup_context_json(context_file)
        # Small delay to ensure different timestamps
        time.sleep(0.01)

    # Verify
    backup_dir = tmp_path / "backups" / "test-context"
    backup_files = list(backup_dir.glob("context-*.json"))
    assert len(backup_files) == 30


def test_backup_context_json_timestamp_format(tmp_path: Path) -> None:
    """Test that backup files use YYYYMMDD-HHMMSS-mmm timestamp format."""
    # Setup
    context_dir = tmp_path / "contexts" / "test-context"
    context_dir.mkdir(parents=True)
    context_file = context_dir / "context.json"
    context_file.write_text(json.dumps({"name": "test"}), encoding="utf-8")

    # Set backup root to tmp_path
    import os
    os.environ["CHINVEX_BACKUPS_ROOT"] = str(tmp_path / "backups")

    # Execute
    backup_context_json(context_file)

    # Verify
    backup_dir = tmp_path / "backups" / "test-context"
    backup_files = list(backup_dir.glob("context-*.json"))
    assert len(backup_files) == 1

    # Check filename format: context-YYYYMMDD-HHMMSS-mmm.json
    filename = backup_files[0].name
    assert filename.startswith("context-")
    assert filename.endswith(".json")

    # Extract timestamp part (remove "context-" prefix and ".json" suffix)
    timestamp_part = filename[8:-5]  # "context-".len = 8, ".json".len = 5

    # Should be YYYYMMDD-HHMMSS-mmm format (21 characters: 8 + 1 + 6 + 1 + 3)
    assert len(timestamp_part) == 19
    assert timestamp_part[8] == "-"  # Separator between date and time
    assert timestamp_part[15] == "-"  # Separator between time and milliseconds

    # Should be all digits except the separators
    date_part = timestamp_part[:8]
    time_part = timestamp_part[9:15]
    ms_part = timestamp_part[16:19]
    assert date_part.isdigit()
    assert time_part.isdigit()
    assert ms_part.isdigit()
