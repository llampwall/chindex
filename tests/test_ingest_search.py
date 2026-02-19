import json
from pathlib import Path

import pytest

from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, RepoMetadata
from chinvex.ingest import ingest_context
from chinvex.search import search_context
from chinvex.storage import Storage


class FakeProvider:
    """Fake embedding provider that avoids network calls."""
    dimensions = 3
    model_name = "fake-model"
    model = "fake-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.1, 0.2] for _ in texts]


def _make_context(tmp_path: Path, repo: Path) -> ContextConfig:
    db_path = tmp_path / "index" / "hybrid.db"
    chroma_dir = tmp_path / "index" / "chroma"
    return ContextConfig(
        schema_version=2,
        name="TestCtx",
        aliases=[],
        includes=ContextIncludes(
            repos=[RepoMetadata(path=repo, chinvex_depth="full", status="active", tags=[])],
            chat_roots=[],
            codex_session_roots=[],
            note_roots=[]
        ),
        index=ContextIndex(sqlite_path=db_path, chroma_dir=chroma_dir),
        weights={"repo": 1.0},
        ollama=OllamaConfig(base_url="http://127.0.0.1:11434", embed_model="fake-model"),
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def _write_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello world from repo", encoding="utf-8")
    return repo


def test_ingest_creates_db_and_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path)
    ctx = _make_context(tmp_path, repo)

    monkeypatch.setattr("chinvex.embedding_providers.get_provider", lambda *a, **kw: FakeProvider())

    result = ingest_context(ctx)
    assert len(result.new_doc_ids) > 0 or len(result.updated_doc_ids) > 0 or result.stats.get("documents", 0) > 0

    db_path = ctx.index.sqlite_path
    storage = Storage(db_path)
    storage.ensure_schema()
    cur = storage.conn.execute("SELECT COUNT(*) AS c FROM documents")
    assert cur.fetchone()["c"] > 0
    cur = storage.conn.execute("SELECT COUNT(*) AS c FROM chunks")
    assert cur.fetchone()["c"] > 0
    cur = storage.conn.execute("SELECT COUNT(*) AS c FROM chunks_fts")
    assert cur.fetchone()["c"] > 0
    Storage.force_close_global_connection()

    chroma_dir = ctx.index.chroma_dir
    assert chroma_dir.exists()
    from chinvex.vectors import VectorStore
    vectors = VectorStore(chroma_dir)
    assert vectors.count() > 0


def test_search_returns_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path)
    ctx = _make_context(tmp_path, repo)

    monkeypatch.setattr("chinvex.embedding_providers.get_provider", lambda *a, **kw: FakeProvider())

    ingest_context(ctx)
    Storage.force_close_global_connection()

    results = search_context(ctx, "hello", k=5)
    assert results
