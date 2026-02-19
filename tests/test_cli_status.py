# tests/test_cli_status.py
import pytest
from pathlib import Path
from chinvex.cli_status import format_status_output, ContextStatus


def test_format_status_output_healthy():
    """Should format healthy context with green indicator"""
    contexts = [
        ContextStatus(
            name="Chinvex",
            chunks=1234,
            last_sync="2026-01-29T10:00:00Z",
            is_stale=False,
            hours_since_sync=2.5,
            watcher_running=True,
            embedding_provider="openai"
        ),
        ContextStatus(
            name="Streamside",
            chunks=567,
            last_sync="2026-01-29T11:30:00Z",
            is_stale=False,
            hours_since_sync=1.0,
            watcher_running=True,
            embedding_provider="openai"
        )
    ]

    output = format_status_output(contexts, watcher_running=True)

    assert "Chinvex" in output
    assert "1234" in output
    assert "[OK]" in output  # Healthy indicator
    assert "Watcher: Running" in output


def test_format_status_output_stale():
    """Should format stale context with warning indicator"""
    contexts = [
        ContextStatus(
            name="Godex",
            chunks=890,
            last_sync="2026-01-28T05:00:00Z",
            is_stale=True,
            hours_since_sync=31.0,
            watcher_running=False,
            embedding_provider="openai"
        )
    ]

    output = format_status_output(contexts, watcher_running=False)

    assert "Godex" in output
    assert "[STALE]" in output  # Stale indicator
    assert "31h ago" in output
    assert "Watcher: Stopped" in output


def test_format_status_reads_global_status_md(tmp_path: Path):
    """read_global_status generates live status from STATUS.json files (not GLOBAL_STATUS.md)."""
    import json
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create two context directories with STATUS.json files
    for ctx_name in ("Chinvex", "Godex"):
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()
        status_data = {
            "chunks": 1234 if ctx_name == "Chinvex" else 890,
            "last_sync": "2026-02-18T10:00:00Z",
            "freshness": {
                "is_stale": ctx_name == "Godex",
                "hours_since_sync": 2.5 if ctx_name == "Chinvex" else 31.0
            }
        }
        (ctx_dir / "STATUS.json").write_text(json.dumps(status_data), encoding="utf-8")

    from chinvex.cli_status import read_global_status
    output = read_global_status(contexts_root)

    assert "Chinvex" in output
    assert "Godex" in output
