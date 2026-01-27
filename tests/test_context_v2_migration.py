# tests/test_context_v2_migration.py
import json
from pathlib import Path
from chinvex.context import load_context, ContextConfig


def test_context_v1_to_v2_migration(tmp_path: Path) -> None:
    """Test auto-upgrade from schema v1 to v2."""
    # Create v1 context structure: contexts_root/TestCtx/context.json
    contexts_root = tmp_path / "contexts"
    context_dir = contexts_root / "TestCtx"
    context_dir.mkdir(parents=True, exist_ok=True)

    v1_data = {
        "schema_version": 1,
        "name": "TestCtx",
        "aliases": [],
        "includes": {
            "repos": [],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": str(tmp_path / "test.db"),
            "chroma_dir": str(tmp_path / "chroma/")
        },
        "weights": {"repo": 1.0},
        "ollama": {
            "base_url": "http://skynet:11434",
            "embed_model": "mxbai-embed-large"
        },
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }

    context_file = context_dir / "context.json"
    context_file.write_text(json.dumps(v1_data), encoding="utf-8")

    # Load context (should auto-upgrade)
    context = load_context("TestCtx", contexts_root)

    # Verify v2 fields added
    assert context.schema_version == 2
    assert hasattr(context, 'codex_appserver')
    assert hasattr(context, 'ranking')
    assert context.codex_appserver.enabled is False
    assert context.ranking.recency_enabled is True

    # Verify file was updated
    saved_data = json.loads(context_file.read_text(encoding="utf-8"))
    assert saved_data["schema_version"] == 2
    assert "codex_appserver" in saved_data
    assert "ranking" in saved_data
