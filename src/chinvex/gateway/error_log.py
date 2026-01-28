"""Rotating error logger for gateway exceptions."""

import json
import traceback
from pathlib import Path
from datetime import datetime


class RotatingErrorLogger:
    """
    Rotating JSONL error logger with size-based rotation.

    Keeps last N files, rotates when current file exceeds max size.
    """

    def __init__(self, base_path: str, max_size_mb: int = 50, max_files: int = 5):
        """
        Initialize rotating error logger.

        Args:
            base_path: Base path for error logs (e.g., "P:/ai_memory/gateway_errors.jsonl")
            max_size_mb: Max size per file before rotation (default 50MB)
            max_files: Max number of rotated files to keep (default 5)
        """
        self.base_path = Path(base_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, request_id: str, method: str, path: str, error: Exception, timestamp: float):
        """
        Log an error with automatic rotation.

        Args:
            request_id: Request identifier
            method: HTTP method
            path: Request path
            error: Exception object
            timestamp: Unix timestamp
        """
        # Check if rotation needed
        if self.base_path.exists() and self.base_path.stat().st_size > self.max_size_bytes:
            self._rotate()

        # Append error entry
        entry = {
            "request_id": request_id,
            "timestamp": timestamp,
            "iso_ts": datetime.utcfromtimestamp(timestamp).isoformat() + "Z",
            "method": method,
            "path": path,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc()
        }

        try:
            with open(self.base_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Fail silently - don't break the gateway if logging fails
            pass

    def _rotate(self):
        """Rotate log files, keeping last N."""
        # Rename pattern: gateway_errors.jsonl -> gateway_errors.1.jsonl -> gateway_errors.2.jsonl
        base_stem = self.base_path.stem  # e.g., "gateway_errors"
        base_suffix = self.base_path.suffix  # e.g., ".jsonl"
        base_dir = self.base_path.parent

        # Delete oldest file if we're at max
        oldest = base_dir / f"{base_stem}.{self.max_files}{base_suffix}"
        if oldest.exists():
            oldest.unlink()

        # Shift existing rotated files
        for i in range(self.max_files - 1, 0, -1):
            old_file = base_dir / f"{base_stem}.{i}{base_suffix}"
            new_file = base_dir / f"{base_stem}.{i + 1}{base_suffix}"
            if old_file.exists():
                old_file.rename(new_file)

        # Rotate current file to .1
        if self.base_path.exists():
            rotated = base_dir / f"{base_stem}.1{base_suffix}"
            self.base_path.rename(rotated)
