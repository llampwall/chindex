# tests/sync/test_watcher.py
import pytest
import time
from pathlib import Path
from chinvex.sync.watcher import ChangeAccumulator


def test_accumulator_starts_empty():
    """New accumulator should have no changes"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)
    assert len(acc.get_changes()) == 0


def test_accumulator_adds_file_changes(tmp_path: Path):
    """Adding file changes should accumulate paths"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)

    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    acc.add_change(file1)
    acc.add_change(file2)

    changes = acc.get_changes()
    assert len(changes) == 2
    assert file1 in changes
    assert file2 in changes


def test_accumulator_deduplicates_paths(tmp_path: Path):
    """Same path added multiple times should appear once"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)

    file1 = tmp_path / "file1.txt"

    acc.add_change(file1)
    acc.add_change(file1)
    acc.add_change(file1)

    changes = acc.get_changes()
    assert len(changes) == 1


def test_accumulator_clears_after_get():
    """get_and_clear() should return changes and reset"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)

    acc.add_change(Path("file1.txt"))
    acc.add_change(Path("file2.txt"))

    changes = acc.get_and_clear()
    assert len(changes) == 2

    # Should be empty after clear
    assert len(acc.get_changes()) == 0


def test_accumulator_respects_max_paths():
    """Should track when max_paths exceeded"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=3)

    for i in range(5):
        acc.add_change(Path(f"file{i}.txt"))

    assert acc.is_over_limit()


def test_accumulator_debounce_timer():
    """Should track time since last change"""
    acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=500)

    acc.add_change(Path("file1.txt"))

    # Immediately after, not ready
    assert not acc.is_ready()

    # After debounce period, should be ready
    time.sleep(0.15)
    assert acc.is_ready()


def test_accumulator_resets_timer_on_new_change():
    """Adding new change should reset debounce timer"""
    acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=500)

    acc.add_change(Path("file1.txt"))
    time.sleep(0.08)  # Almost ready

    # Add another change - should reset timer
    acc.add_change(Path("file2.txt"))

    # Should not be ready yet
    assert not acc.is_ready()


def test_accumulator_max_debounce_cap():
    """Should force ingest after 5 minutes even if changes keep coming"""
    acc = ChangeAccumulator(debounce_seconds=30, max_paths=500)

    # Simulate the first change
    acc.add_change(Path("file1.txt"))

    # Mock time to simulate 5+ minutes passing
    original_time = acc._first_change_time
    acc._first_change_time = original_time - 301  # 5min + 1s ago

    # Even though normal debounce hasn't elapsed, should be ready due to cap
    assert acc.is_ready()
