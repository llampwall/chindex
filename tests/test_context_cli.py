from pathlib import Path
import json
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_context_create_success(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, ["context", "create", "TestContext"])
    assert result.exit_code == 0
    assert "Created context: TestContext" in result.stdout

    # Verify structure
    ctx_file = contexts_root / "TestContext" / "context.json"
    assert ctx_file.exists()
    data = json.loads(ctx_file.read_text())
    assert data["name"] == "TestContext"
    assert data["schema_version"] == 1

    index_dir = indexes_root / "TestContext"
    assert index_dir.exists()
    assert (index_dir / "hybrid.db").exists()


def test_context_create_already_exists(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "TestContext"])
    result = runner.invoke(app, ["context", "create", "TestContext"])

    assert result.exit_code == 1
    assert "already exists" in result.stdout


def test_context_create_invalid_name(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, ["context", "create", "Test/Invalid"])
    assert result.exit_code == 2
    assert "invalid" in result.stdout.lower()
