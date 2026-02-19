"""Test --rebuild-index flag for provider switching."""

import sys
import pytest
from pathlib import Path
from datetime import datetime
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, RepoMetadata
from chinvex.ingest import ingest_context
from chinvex.index_meta import read_index_meta
from chinvex.storage import Storage

# ChromaDB holds file handles on Windows, preventing shutil.rmtree during rebuild
_WINDOWS_CHROMA_SKIP = pytest.mark.skipif(
    sys.platform == "win32",
    reason="ChromaDB file locking prevents index rebuild on Windows"
)


@_WINDOWS_CHROMA_SKIP
def test_rebuild_index_on_provider_switch(tmp_path: Path, monkeypatch):
    """Test that --rebuild-index clears and rebuilds index when switching providers."""
    # Mock OllamaEmbedder to return fake embeddings
    class FakeEmbedder:
        def __init__(self, host: str, model: str, fallback_host: str | None = None):
            self.host = host
            self._model = model
            self.fallback_host = fallback_host

        def embed(self, texts: list[str]) -> list[list[float]]:
            # Return 768D for Ollama, 1536D for OpenAI (mocked)
            return [[0.1] * 768 for _ in texts]

        @property
        def model(self):
            return self._model

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    # Mock providers to return correct dimensions
    # Provider class names must produce correct names when .replace("Provider","").lower() is applied
    # OllamaProvider -> "ollama", OpenAIProvider -> "openai"
    class OllamaProvider:
        dimensions = 1024
        model_name = "mxbai-embed-large"
        model = "mxbai-embed-large"
        def embed(self, texts): return [[0.1] * 1024 for _ in texts]

    class OpenAIProvider:
        dimensions = 1536
        model_name = "text-embedding-3-small"
        model = "text-embedding-3-small"
        def embed(self, texts): return [[0.1] * 1536 for _ in texts]

    def fake_get_provider(cli_provider, context_config, env_provider, ollama_host):
        if cli_provider == "openai" or env_provider == "openai":
            return OpenAIProvider()
        return OllamaProvider()

    monkeypatch.setattr("chinvex.embedding_providers.get_provider", fake_get_provider)

    # Setup context
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "test.py").write_text("def hello(): pass")

    index_dir = tmp_path / "index"
    db_path = index_dir / "hybrid.db"
    chroma_dir = index_dir / "chroma"

    now = datetime.now()
    ctx = ContextConfig(
        schema_version=2,
        name="test",
        aliases=[],
        includes=ContextIncludes(
            repos=[RepoMetadata(path=repo_dir, chinvex_depth="full", status="active", tags=[])],
            chat_roots=[],
            codex_session_roots=[],
            note_roots=[]
        ),
        index=ContextIndex(
            sqlite_path=db_path,
            chroma_dir=chroma_dir
        ),
        ollama=OllamaConfig(
            base_url="http://127.0.0.1:11434",
            embed_model="mxbai-embed-large"
        ),
        weights={"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        created_at=now.isoformat(),
        updated_at=now.isoformat()
    )

    # Initial ingest with Ollama
    result = ingest_context(ctx, embed_provider="ollama")
    assert len(result.error_doc_ids) == 0

    # Verify meta.json created with Ollama
    meta = read_index_meta(index_dir / "meta.json")
    assert meta is not None
    assert meta.embedding_provider == "ollama"
    assert meta.embedding_dimensions == 1024

    # Verify SQLite and Chroma exist
    assert db_path.exists()
    assert chroma_dir.exists()

    # Get initial chunk count
    storage = Storage(db_path)
    initial_chunks = storage.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert initial_chunks > 0
    Storage.force_close_global_connection()

    # Try switching to OpenAI without --rebuild-index (should fail)
    with pytest.raises(RuntimeError, match="Dimension mismatch.*Use --rebuild-index"):
        ingest_context(ctx, embed_provider="openai", rebuild_index=False)

    # Verify index still exists after failed attempt
    assert db_path.exists()
    assert chroma_dir.exists()

    # Switch to OpenAI with --rebuild-index (should succeed)
    result = ingest_context(ctx, embed_provider="openai", rebuild_index=True)
    assert len(result.error_doc_ids) == 0

    # Verify meta.json updated with OpenAI
    meta = read_index_meta(index_dir / "meta.json")
    assert meta is not None
    assert meta.embedding_provider == "openai"
    assert meta.embedding_dimensions == 1536

    # Verify index was rebuilt (files recreated)
    assert db_path.exists()
    assert chroma_dir.exists()

    # Verify chunks were re-ingested
    storage = Storage(db_path)
    rebuilt_chunks = storage.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert rebuilt_chunks > 0
    assert rebuilt_chunks == initial_chunks  # Same content, just rebuilt
    Storage.force_close_global_connection()


@_WINDOWS_CHROMA_SKIP
def test_rebuild_index_clears_old_data(tmp_path: Path, monkeypatch):
    """Test that --rebuild-index actually clears old chunks and embeddings when switching providers."""
    # Mock embedder
    class FakeEmbedder:
        def __init__(self, host: str, model: str, fallback_host: str | None = None):
            self._model = model
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1024 for _ in texts]

        @property
        def model(self):
            return self._model

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    # Use two providers with different dimensions to trigger rebuild
    # Provider class names are used to derive provider name via .replace("Provider","").lower()
    class OllamaProvider:
        dimensions = 1024
        model_name = "mxbai-embed-large"
        model = "mxbai-embed-large"
        def embed(self, texts): return [[0.1] * 1024 for _ in texts]

    class OpenAIProvider:
        dimensions = 1536
        model_name = "text-embedding-3-small"
        model = "text-embedding-3-small"
        def embed(self, texts): return [[0.1] * 1536 for _ in texts]

    provider_switch = {"use_openai": False}

    def fake_get_provider(*args, **kwargs):
        if provider_switch["use_openai"]:
            return OpenAIProvider()
        return OllamaProvider()

    monkeypatch.setattr("chinvex.embedding_providers.get_provider", fake_get_provider)

    # Setup context with initial content
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "file1.py").write_text("def hello(): pass")

    index_dir = tmp_path / "index"
    db_path = index_dir / "hybrid.db"
    chroma_dir = index_dir / "chroma"

    now = datetime.now()
    ctx = ContextConfig(
        schema_version=2,
        name="test",
        aliases=[],
        includes=ContextIncludes(
            repos=[RepoMetadata(path=repo_dir, chinvex_depth="full", status="active", tags=[])],
            chat_roots=[],
            codex_session_roots=[],
            note_roots=[]
        ),
        index=ContextIndex(
            sqlite_path=db_path,
            chroma_dir=chroma_dir
        ),
        ollama=OllamaConfig(
            base_url="http://127.0.0.1:11434",
            embed_model="mxbai-embed-large"
        ),
        weights={"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        created_at=now.isoformat(),
        updated_at=now.isoformat()
    )

    # Initial ingest with OllamaProvider (1024D)
    result = ingest_context(ctx)
    assert len(result.error_doc_ids) == 0

    # Get initial chunk IDs
    storage = Storage(db_path)
    initial_chunk_ids = set(row[0] for row in storage.conn.execute("SELECT chunk_id FROM chunks").fetchall())
    assert len(initial_chunk_ids) > 0
    Storage.force_close_global_connection()

    # Delete original file and add new file
    (repo_dir / "file1.py").unlink()
    (repo_dir / "file2.py").write_text("def world(): pass")

    # Switch to OpenAI provider and rebuild index (dimension mismatch triggers rebuild)
    provider_switch["use_openai"] = True
    result = ingest_context(ctx, rebuild_index=True)
    assert len(result.error_doc_ids) == 0

    # Get new chunk IDs
    storage = Storage(db_path)
    new_chunk_ids = set(row[0] for row in storage.conn.execute("SELECT chunk_id FROM chunks").fetchall())
    assert len(new_chunk_ids) > 0
    Storage.force_close_global_connection()

    # Verify old chunks are gone (chunk IDs should be different)
    assert initial_chunk_ids != new_chunk_ids

    # Verify only file2.py is in the index (file1.py chunks cleared by rebuild)
    storage = Storage(db_path)
    doc_ids = [row[0] for row in storage.conn.execute("SELECT DISTINCT doc_id FROM chunks").fetchall()]
    assert len(doc_ids) == 1
    # Check that the doc_id contains file2 (not file1)
    doc_meta = storage.conn.execute("SELECT meta_json FROM chunks WHERE doc_id = ?", (doc_ids[0],)).fetchone()[0]
    assert "file2" in doc_meta
    Storage.force_close_global_connection()
