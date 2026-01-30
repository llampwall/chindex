# tests/hooks/test_resolver.py
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.hooks.resolver import resolve_python_path


def test_resolve_finds_venv_python(tmp_path: Path):
    """Should prefer repo-local venv Python"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Create mock venv
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("mock python")

    # Mock subprocess to simulate python --version
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="Python 3.12.0")

        result = resolve_python_path(repo_root)

        assert result == str(venv_python)


def test_resolve_fallback_to_py_launcher(tmp_path: Path):
    """If no venv, should try py -3"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # No venv exists
    with patch('subprocess.run') as mock_run:
        # py -3: --version check, then -m chinvex.cli check
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Python 3.12.0"),  # py -3 --version
            Mock(returncode=0, stdout="")  # py -3 -m chinvex.cli --help
        ]

        with patch('shutil.which', return_value="C:\\Windows\\py.exe"):
            result = resolve_python_path(repo_root)

            assert result == "py -3"


def test_resolve_uses_env_override(tmp_path: Path, monkeypatch):
    """CHINVEX_PYTHON should override"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    custom_python = "C:\\CustomPython\\python.exe"
    monkeypatch.setenv("CHINVEX_PYTHON", custom_python)

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="Python 3.12.0")

        result = resolve_python_path(repo_root)

        assert result == custom_python


def test_resolve_validates_chinvex_installed(tmp_path: Path):
    """Should verify chinvex module is importable"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("mock")

    with patch('subprocess.run') as mock_run:
        # First call: python --version (passes)
        # Second call: python -m chinvex.cli --help (fails - chinvex not installed)
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Python 3.12.0"),
            Mock(returncode=1)  # chinvex not installed
        ]

        # Should skip this python and try next option
        with patch('shutil.which', return_value="C:\\Windows\\py.exe"):
            mock_run.side_effect = [
                Mock(returncode=0, stdout="Python 3.12.0"),
                Mock(returncode=1),  # venv chinvex check fails
                Mock(returncode=0, stdout="Python 3.12.0"),
                Mock(returncode=0)   # py -3 chinvex check succeeds
            ]

            result = resolve_python_path(repo_root)

            assert result == "py -3"
