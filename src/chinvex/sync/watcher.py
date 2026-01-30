"""File watcher with debounce and change accumulation."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Set

log = logging.getLogger(__name__)


class ChangeAccumulator:
    """
    Accumulates file changes with debounce logic.

    Debounce semantics:
    - Wait debounce_seconds after last change before considering ready
    - Track all changed paths (deduplicated)
    - Flag if path count exceeds max_paths (triggers full ingest fallback)
    - Force ingest after 5 minutes even if changes keep coming (max debounce cap)
    """

    MAX_DEBOUNCE_SECONDS = 300  # 5 minutes cap

    def __init__(self, debounce_seconds: float, max_paths: int):
        """
        Args:
            debounce_seconds: Seconds of quiet before ready
            max_paths: Max paths before flagging full ingest needed
        """
        self.debounce_seconds = debounce_seconds
        self.max_paths = max_paths

        self._changes: Set[Path] = set()
        self._last_change_time: float | None = None
        self._first_change_time: float | None = None  # Track when accumulation started

    def add_change(self, path: Path) -> None:
        """
        Add a changed file path.

        Resets debounce timer.
        """
        self._changes.add(path)
        self._last_change_time = time.time()

        # Track first change time for max debounce cap
        if self._first_change_time is None:
            self._first_change_time = time.time()

        log.debug(f"Change recorded: {path} (total: {len(self._changes)})")

    def get_changes(self) -> list[Path]:
        """Get accumulated changes without clearing."""
        return list(self._changes)

    def get_and_clear(self) -> list[Path]:
        """Get accumulated changes and clear accumulator."""
        changes = list(self._changes)
        self._changes.clear()
        self._last_change_time = None
        self._first_change_time = None
        return changes

    def is_ready(self) -> bool:
        """
        Check if enough quiet time has passed since last change.

        Returns True if either:
        - Debounce period elapsed and changes exist
        - Total debounce time exceeds 5 minutes (force ingest)

        Returns:
            True if debounce period elapsed and changes exist
        """
        if not self._changes:
            return False

        if self._last_change_time is None:
            return False

        # Check if max debounce cap exceeded (5 min total)
        if self._first_change_time is not None:
            total_time = time.time() - self._first_change_time
            if total_time >= self.MAX_DEBOUNCE_SECONDS:
                log.info(f"Max debounce cap ({self.MAX_DEBOUNCE_SECONDS}s) exceeded - forcing ingest")
                return True

        # Normal debounce check
        elapsed = time.time() - self._last_change_time
        return elapsed >= self.debounce_seconds

    def is_over_limit(self) -> bool:
        """Check if accumulated paths exceed max_paths."""
        return len(self._changes) > self.max_paths
