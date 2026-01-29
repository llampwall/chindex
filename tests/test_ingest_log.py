import pytest
from pathlib import Path
from chinvex.ingest_log import log_run_start, log_run_end, read_ingest_runs


def test_log_run_start(tmp_path):
    """Test logging run start."""
    log_path = tmp_path / "ingest_runs.jsonl"
    run_id = "test-run-123"

    log_run_start(log_path, run_id, sources=["repo1", "repo2"])

    assert log_path.exists()
    runs = read_ingest_runs(log_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "started"


def test_log_run_end_success(tmp_path):
    """Test logging successful run end."""
    log_path = tmp_path / "ingest_runs.jsonl"
    run_id = "test-run-123"

    log_run_start(log_path, run_id, sources=["repo1"])
    log_run_end(
        log_path,
        run_id,
        status="succeeded",
        docs_seen=100,
        docs_changed=10,
        chunks_new=50,
        chunks_updated=20
    )

    runs = read_ingest_runs(log_path)
    assert len(runs) == 2
    assert runs[1]["status"] == "succeeded"
    assert runs[1]["docs_seen"] == 100


def test_log_run_end_failure(tmp_path):
    """Test logging failed run end."""
    log_path = tmp_path / "ingest_runs.jsonl"
    run_id = "test-run-123"

    log_run_start(log_path, run_id, sources=["repo1"])
    log_run_end(
        log_path,
        run_id,
        status="failed",
        error="OpenAI rate limit exceeded"
    )

    runs = read_ingest_runs(log_path)
    assert len(runs) == 2
    assert runs[1]["status"] == "failed"
    assert "rate limit" in runs[1]["error"]


def test_read_completed_runs_only(tmp_path):
    """Test filtering for completed runs only."""
    log_path = tmp_path / "ingest_runs.jsonl"

    # Run 1: completed
    log_run_start(log_path, "run1", sources=["repo1"])
    log_run_end(log_path, "run1", status="succeeded", docs_seen=10)

    # Run 2: started but not completed (crash)
    log_run_start(log_path, "run2", sources=["repo1"])

    # Run 3: completed
    log_run_start(log_path, "run3", sources=["repo1"])
    log_run_end(log_path, "run3", status="succeeded", docs_seen=20)

    completed = read_ingest_runs(log_path, completed_only=True)
    assert len(completed) == 2
    assert completed[0]["run_id"] == "run1"
    assert completed[1]["run_id"] == "run3"
