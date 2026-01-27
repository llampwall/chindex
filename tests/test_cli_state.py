# tests/test_cli_state.py
from typer.testing import CliRunner
from chinvex.cli import app

def test_state_generate_command():
    """Test chinvex state generate command."""
    runner = CliRunner()
    result = runner.invoke(app, ['state', 'generate', '--context', 'TestContext'])

    # Command should exist (even if it fails due to missing context)
    # Exit code 0 = success, 1 = error but command exists, 2 = command not found
    assert result.exit_code != 2, "state generate command not found"

def test_state_show_command():
    """Test chinvex state show command."""
    runner = CliRunner()
    result = runner.invoke(app, ['state', 'show', '--context', 'TestContext'])

    # Command should exist (even if it fails due to missing file)
    assert result.exit_code != 2, "state show command not found"
