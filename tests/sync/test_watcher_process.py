# tests/sync/test_watcher_process.py
import pytest
import time
from pathlib import Path
from chinvex.sync.process import WatcherProcess


def test_watcher_creates_observers(tmp_path: Path):
    """Watcher should create file observers for sources"""
    # Create mock sources
    repo1 = tmp_path / "repo1"
    repo1.mkdir()

    from chinvex.sync.discovery import WatchSource
    sources = [
        WatchSource(context_name="Ctx1", source_type="repo", path=repo1)
    ]

    watcher = WatcherProcess(
        sources=sources,
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Should have created observer
    assert len(watcher._observers) == 1


def test_watcher_accumulates_changes(tmp_path: Path):
    """File changes should accumulate in context accumulators"""
    repo1 = tmp_path / "repo1"
    repo1.mkdir()

    from chinvex.sync.discovery import WatchSource
    sources = [
        WatchSource(context_name="Ctx1", source_type="repo", path=repo1)
    ]

    watcher = WatcherProcess(
        sources=sources,
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Simulate file change event
    test_file = repo1 / "test.txt"
    test_file.write_text("content")

    # Manually trigger change handler
    watcher._on_file_changed(str(test_file), "Ctx1")

    # Should have accumulated change
    assert "Ctx1" in watcher._accumulators
    changes = watcher._accumulators["Ctx1"].get_changes()
    assert len(changes) > 0


def test_watcher_respects_exclude_patterns(tmp_path: Path):
    """Excluded files should not trigger accumulation"""
    repo1 = tmp_path / "repo1"
    repo1.mkdir()

    from chinvex.sync.discovery import WatchSource
    sources = [
        WatchSource(context_name="Ctx1", source_type="repo", path=repo1)
    ]

    watcher = WatcherProcess(
        sources=sources,
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Try to add excluded file
    excluded_file = repo1 / ".git" / "config"
    watcher._on_file_changed(str(excluded_file), "Ctx1")

    # Should NOT have accumulated
    if "Ctx1" in watcher._accumulators:
        changes = watcher._accumulators["Ctx1"].get_changes()
        assert len(changes) == 0


def test_watcher_writes_heartbeat(tmp_path: Path):
    """Watcher should write heartbeat periodically"""
    from chinvex.sync.discovery import WatchSource

    watcher = WatcherProcess(
        sources=[],
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Manually trigger heartbeat write
    watcher._write_heartbeat()

    heartbeat_file = tmp_path / "state" / "sync_heartbeat.json"
    assert heartbeat_file.exists()
