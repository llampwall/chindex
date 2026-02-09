"""Integration test for context.json backup functionality."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chinvex.context_cli import create_context_if_missing, sync_metadata_from_strap


def test_backup_on_context_creation(tmp_path: Path) -> None:
    """Test that backup is created when updating an existing context."""
    # Setup
    os.environ["CHINVEX_CONTEXTS_ROOT"] = str(tmp_path / "contexts")
    os.environ["CHINVEX_INDEXES_ROOT"] = str(tmp_path / "indexes")
    os.environ["CHINVEX_BACKUPS_ROOT"] = str(tmp_path / "backups")

    # Create initial context
    create_context_if_missing(
        name="test-context",
        contexts_root=tmp_path / "contexts",
        repos=[{
            "path": "P:/software/test",
            "chinvex_depth": "full",
            "status": "active",
            "tags": ["python"]
        }]
    )

    # No backup yet (first creation)
    backup_dir = tmp_path / "backups" / "test-context"
    assert not backup_dir.exists() or len(list(backup_dir.glob("context-*.json"))) == 0

    # Update context with new repo (should trigger backup)
    create_context_if_missing(
        name="test-context",
        contexts_root=tmp_path / "contexts",
        repos=[{
            "path": "P:/software/another",
            "chinvex_depth": "light",
            "status": "stable",
            "tags": ["js"]
        }]
    )

    # Verify backup was created
    assert backup_dir.exists()
    backup_files = list(backup_dir.glob("context-*.json"))
    assert len(backup_files) == 1

    # Verify backup content matches pre-update state
    backup_content = json.loads(backup_files[0].read_text(encoding="utf-8"))
    assert len(backup_content["includes"]["repos"]) == 1
    assert backup_content["includes"]["repos"][0]["path"] == "P:\\software\\test"


def test_backup_on_sync_metadata(tmp_path: Path) -> None:
    """Test that backup is created when syncing metadata from strap."""
    # Setup
    os.environ["CHINVEX_CONTEXTS_ROOT"] = str(tmp_path / "contexts")
    os.environ["CHINVEX_INDEXES_ROOT"] = str(tmp_path / "indexes")
    os.environ["CHINVEX_BACKUPS_ROOT"] = str(tmp_path / "backups")

    # Create context
    context_dir = tmp_path / "contexts" / "test-context"
    context_dir.mkdir(parents=True)
    context_file = context_dir / "context.json"
    context_data = {
        "schema_version": 1,
        "name": "test-context",
        "aliases": [],
        "includes": {
            "repos": [{
                "path": "P:/software/chinvex",
                "chinvex_depth": "full",
                "status": "active",
                "tags": []
            }],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": str(tmp_path / "indexes" / "test-context" / "hybrid.db"),
            "chroma_dir": str(tmp_path / "indexes" / "test-context" / "chroma")
        },
        "weights": {"repo": 1.0},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    }
    context_file.write_text(json.dumps(context_data, indent=2), encoding="utf-8")

    # Create fake registry
    registry_file = tmp_path / "registry.json"
    registry_data = {
        "repos": [{
            "repoPath": "P:/software/chinvex",
            "chinvex_depth": "light",
            "status": "stable",
            "tags": ["python", "search"]
        }]
    }
    registry_file.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    # Sync metadata (should create backup)
    sync_metadata_from_strap(
        context_name="test-context",
        registry_path=registry_file
    )

    # Verify backup was created
    backup_dir = tmp_path / "backups" / "test-context"
    assert backup_dir.exists()
    backup_files = list(backup_dir.glob("context-*.json"))
    assert len(backup_files) == 1

    # Verify backup content matches pre-sync state
    backup_content = json.loads(backup_files[0].read_text(encoding="utf-8"))
    assert backup_content["includes"]["repos"][0]["chinvex_depth"] == "full"
    assert backup_content["includes"]["repos"][0]["status"] == "active"
    assert backup_content["includes"]["repos"][0]["tags"] == []

    # Verify actual context was updated
    updated_content = json.loads(context_file.read_text(encoding="utf-8"))
    assert updated_content["includes"]["repos"][0]["chinvex_depth"] == "light"
    assert updated_content["includes"]["repos"][0]["status"] == "stable"
    assert updated_content["includes"]["repos"][0]["tags"] == ["python", "search"]
