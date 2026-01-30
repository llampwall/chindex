"""Tests for Task Scheduler registration."""
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.bootstrap.scheduler import (
    register_sweep_task,
    unregister_sweep_task,
    register_login_trigger_task,
    unregister_login_trigger_task,
)


def test_register_creates_task_xml():
    """Should generate valid Task Scheduler XML"""
    from chinvex.bootstrap.scheduler import _generate_task_xml

    contexts_root = Path("P:/ai_memory/contexts")
    script_path = Path("C:/Code/chinvex/scripts/scheduled_sweep.ps1")
    ntfy_topic = "chinvex-alerts"

    xml = _generate_task_xml(script_path, contexts_root, ntfy_topic)

    assert "<Task" in xml
    assert "scheduled_sweep.ps1" in xml
    assert str(contexts_root) in xml
    assert "PT30M" in xml  # 30 minute interval


def test_register_calls_schtasks(tmp_path: Path):
    """Should call schtasks.exe to register task"""
    contexts_root = tmp_path / "contexts"
    script_path = tmp_path / "sweep.ps1"
    script_path.write_text("# mock script")

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        register_sweep_task(script_path, contexts_root, "topic")

        # Should have called schtasks
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Create" in call_args


def test_unregister_removes_task():
    """Should call schtasks to delete task"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        unregister_sweep_task()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Delete" in call_args
        assert "ChinvexSweep" in call_args


def test_register_validates_script_exists(tmp_path: Path):
    """Should fail if script doesn't exist"""
    script_path = tmp_path / "nonexistent.ps1"
    contexts_root = tmp_path / "contexts"

    with pytest.raises(FileNotFoundError):
        register_sweep_task(script_path, contexts_root, "topic")


def test_register_login_trigger():
    """Should register login-trigger task that starts watcher"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        register_login_trigger_task()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Create" in call_args
        assert "ChinvexWatcherStart" in " ".join(call_args)


def test_unregister_login_trigger():
    """Should remove login-trigger task"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        unregister_login_trigger_task()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Delete" in call_args
        assert "ChinvexWatcherStart" in " ".join(call_args)
