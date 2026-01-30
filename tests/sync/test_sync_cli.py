# tests/sync/test_sync_cli.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_sync_start_when_not_running(tmp_path: Path, monkeypatch):
    """sync start should start daemon when not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "start"])

    # Should succeed
    assert result.exit_code == 0
    assert "started" in result.stdout.lower() or "daemon" in result.stdout.lower()

    # Should write PID file
    assert (tmp_path / "sync.pid").exists()


def test_sync_start_when_already_running(tmp_path: Path, monkeypatch):
    """sync start should refuse if already running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start first time
    result1 = runner.invoke(app, ["sync", "start"])
    assert result1.exit_code == 0

    # Try to start again - should fail
    result2 = runner.invoke(app, ["sync", "start"])
    assert result2.exit_code != 0
    assert "already running" in result2.stdout.lower()


def test_sync_stop(tmp_path: Path, monkeypatch):
    """sync stop should stop running daemon"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start daemon
    runner.invoke(app, ["sync", "start"])
    assert (tmp_path / "sync.pid").exists()

    # Stop daemon
    result = runner.invoke(app, ["sync", "stop"])
    assert result.exit_code == 0

    # PID file should be removed
    assert not (tmp_path / "sync.pid").exists()


def test_sync_status_not_running(tmp_path: Path, monkeypatch):
    """sync status should show not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "status"])
    assert result.exit_code == 0
    assert "not running" in result.stdout.lower()


def test_sync_status_running(tmp_path: Path, monkeypatch):
    """sync status should show running state"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start daemon
    runner.invoke(app, ["sync", "start"])

    result = runner.invoke(app, ["sync", "status"])
    assert result.exit_code == 0
    assert "running" in result.stdout.lower()


def test_sync_ensure_running_starts_if_stopped(tmp_path: Path, monkeypatch):
    """ensure-running should start daemon if not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "ensure-running"])
    assert result.exit_code == 0

    # Should have started daemon
    assert (tmp_path / "sync.pid").exists()


def test_sync_ensure_running_noop_if_running(tmp_path: Path, monkeypatch):
    """ensure-running should be no-op if already running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start daemon
    runner.invoke(app, ["sync", "start"])
    pid1 = (tmp_path / "sync.pid").read_text()

    # Ensure running - should not restart
    result = runner.invoke(app, ["sync", "ensure-running"])
    assert result.exit_code == 0

    pid2 = (tmp_path / "sync.pid").read_text()
    assert pid1 == pid2  # Same PID = didn't restart
