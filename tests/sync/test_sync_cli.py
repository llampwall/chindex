# tests/sync/test_sync_cli.py
import pytest
import time
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app
from chinvex.sync.daemon import DaemonManager, DaemonState

runner = CliRunner()


def wait_for_daemon_running(state_dir: Path, timeout: float = 5.0) -> bool:
    """Wait for daemon to reach RUNNING state (with fresh heartbeat)."""
    dm = DaemonManager(state_dir)
    start_time = time.time()

    while time.time() - start_time < timeout:
        if dm.get_state() == DaemonState.RUNNING:
            return True
        time.sleep(0.1)

    return False


def test_sync_start_when_not_running(tmp_path: Path, monkeypatch):
    """sync start should start daemon when not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "start"])

    # Should succeed
    assert result.exit_code == 0
    assert "started" in result.stdout.lower() or "daemon" in result.stdout.lower()

    # Should write PID file
    assert (tmp_path / "sync.pid").exists()


@pytest.mark.skip(reason="Integration test - requires full daemon E2E setup")
def test_sync_start_when_already_running(tmp_path: Path, monkeypatch):
    """sync start should refuse if already running"""
    state_dir = tmp_path / "state"
    contexts_root = tmp_path / "contexts"
    state_dir.mkdir()
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Start first time
    result1 = runner.invoke(app, ["sync", "start"])
    assert result1.exit_code == 0

    # Wait for daemon to be fully running
    assert wait_for_daemon_running(state_dir), "Daemon failed to start"

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


@pytest.mark.skip(reason="Integration test - requires full daemon E2E setup")
def test_sync_status_running(tmp_path: Path, monkeypatch):
    """sync status should show running state"""
    state_dir = tmp_path / "state"
    contexts_root = tmp_path / "contexts"
    state_dir.mkdir()
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Start daemon
    runner.invoke(app, ["sync", "start"])

    # Wait for daemon to be fully running
    assert wait_for_daemon_running(state_dir), "Daemon failed to start"

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


@pytest.mark.skip(reason="Integration test - requires full daemon E2E setup")
def test_sync_ensure_running_noop_if_running(tmp_path: Path, monkeypatch):
    """ensure-running should be no-op if already running"""
    state_dir = tmp_path / "state"
    contexts_root = tmp_path / "contexts"
    state_dir.mkdir()
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Start daemon
    runner.invoke(app, ["sync", "start"])

    # Wait for daemon to be fully running
    assert wait_for_daemon_running(state_dir), "Daemon failed to start"

    pid1 = (state_dir / "sync.pid").read_text()

    # Ensure running - should not restart
    result = runner.invoke(app, ["sync", "ensure-running"])
    assert result.exit_code == 0

    pid2 = (state_dir / "sync.pid").read_text()
    assert pid1 == pid2  # Same PID = didn't restart
