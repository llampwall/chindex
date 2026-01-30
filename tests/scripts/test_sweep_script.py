"""Tests for PowerShell sweep script."""
import pytest
import subprocess
from pathlib import Path


def test_sweep_script_exists():
    """Sweep script should exist"""
    script_path = Path("scripts/scheduled_sweep.ps1")
    assert script_path.exists()


def test_sweep_script_syntax_valid():
    """PowerShell script should have valid syntax"""
    script_path = Path("scripts/scheduled_sweep.ps1")

    # Test PowerShell syntax
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path), "-WhatIf"],
        capture_output=True,
        text=True
    )

    # Should not have syntax errors
    assert "ParserError" not in result.stderr
    assert "unexpected token" not in result.stderr.lower()


def test_sweep_script_requires_params():
    """Script should require ContextsRoot parameter"""
    script_path = Path("scripts/scheduled_sweep.ps1")

    # Run without required params
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path)],
        capture_output=True,
        text=True
    )

    # Should fail with parameter error
    assert result.returncode != 0
    assert "ContextsRoot" in result.stderr or "parameter" in result.stderr.lower()
