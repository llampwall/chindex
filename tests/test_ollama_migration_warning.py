# tests/test_ollama_migration_warning.py
import pytest
from pathlib import Path
import json
from datetime import datetime, timezone
from chinvex.context import load_context, ContextConfig
from chinvex.search import search_context
import sys
from io import StringIO


def test_ollama_context_shows_migration_warning(tmp_path, monkeypatch, capsys):
    """Searching an Ollama context should print migration warning to stderr."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Create context with Ollama embeddings
    ctx_dir = contexts_root / "OllamaContext"
    ctx_dir.mkdir()
    ctx_index = indexes_root / "OllamaContext"
    ctx_index.mkdir()

    # Create minimal index
    from chinvex.storage import Storage
    db_path = ctx_index / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    ctx_data = {
        "schema_version": 2,
        "name": "OllamaContext",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(db_path), "chroma_dir": str(ctx_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "ollama", "model": "mxbai-embed-large"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_data, indent=2))

    # Load context
    ctx = load_context("OllamaContext", contexts_root)

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        # Attempt search (will fail on embedding, but warning should print first)
        try:
            search_context(
                ctx=ctx,
                query="test query",
                k=5
            )
        except Exception:
            pass  # Expected to fail on actual embedding

        # Get stderr output
        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Verify warning message
    assert "ollama" in stderr_output.lower()
    assert "openai" in stderr_output.lower()
    assert "migrat" in stderr_output.lower() or "consider" in stderr_output.lower()


def test_openai_context_no_migration_warning(tmp_path, capsys):
    """Searching an OpenAI context should not print migration warning."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Create context with OpenAI embeddings
    ctx_dir = contexts_root / "OpenAIContext"
    ctx_dir.mkdir()
    ctx_index = indexes_root / "OpenAIContext"
    ctx_index.mkdir()

    from chinvex.storage import Storage
    db_path = ctx_index / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    ctx_data = {
        "schema_version": 2,
        "name": "OpenAIContext",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(db_path), "chroma_dir": str(ctx_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_data, indent=2))

    ctx = load_context("OpenAIContext", contexts_root)

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        try:
            search_context(ctx=ctx, query="test", k=5)
        except Exception:
            pass

        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Should NOT contain migration warning
    migration_keywords = ["migrat", "consider switching", "consider moving"]
    has_migration_warning = any(kw in stderr_output.lower() for kw in migration_keywords)
    assert not has_migration_warning
