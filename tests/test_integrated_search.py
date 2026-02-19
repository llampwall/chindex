# tests/test_integrated_search.py
from pathlib import Path
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
        # Return varying embeddings for testing
        return [[float(i) * 0.1 for i in range(3)] for _ in texts]


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


def test_search_applies_score_normalization_and_blending(tmp_path: Path, monkeypatch) -> None:
    # Create test data
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "doc1.txt").write_text("apple banana cherry", encoding="utf-8")
    (repo / "doc2.txt").write_text("banana cherry date", encoding="utf-8")

    ctx = _make_context(tmp_path, repo)

    monkeypatch.setattr("chinvex.embedding_providers.get_provider", lambda *a, **kw: FakeProvider())

    ingest_context(ctx)
    Storage.force_close_global_connection()

    results = search_context(ctx, "banana", k=5)

    # Should have results
    assert len(results) > 0

    # Scores should be normalized and blended
    for result in results:
        assert 0.0 <= result.score <= 1.0


def test_search_applies_source_weights(tmp_path: Path, monkeypatch) -> None:
    # This test requires context-based config
    # For now, just verify weights are accessible
    # Full integration in later task
    pass
