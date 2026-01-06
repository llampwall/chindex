import json
from pathlib import Path

import pytest

from chinvex.config import AppConfig, SourceConfig
from chinvex.ingest import ingest
from chinvex.search import search
from chinvex.storage import Storage


class FakeEmbedder:
    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.1, 0.2] for _ in texts]


def _write_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello world from repo", encoding="utf-8")
    return repo


def _write_chat(tmp_path: Path) -> Path:
    chat_dir = tmp_path / "chats"
    chat_dir.mkdir()
    payload = {
        "conversation_id": "conv-1",
        "title": "Test Chat",
        "project": "Twitch",
        "exported_at": "2024-01-01T00:00:00Z",
        "url": "http://example",
        "messages": [
            {"role": "user", "text": "hello chat", "timestamp": None},
            {"role": "assistant", "text": "reply here", "timestamp": None},
        ],
    }
    (chat_dir / "chat.json").write_text(json.dumps(payload), encoding="utf-8")
    return chat_dir


def _make_config(tmp_path: Path, repo: Path, chat_dir: Path) -> AppConfig:
    return AppConfig(
        index_dir=tmp_path / "index",
        ollama_host="http://127.0.0.1:11434",
        embedding_model="mxbai-embed-large",
        sources=(
            SourceConfig(type="repo", name="streamside", path=repo),
            SourceConfig(type="chat", project="Twitch", path=chat_dir),
        ),
    )


def test_ingest_creates_db_and_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path)
    chat_dir = _write_chat(tmp_path)
    config = _make_config(tmp_path, repo, chat_dir)

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    stats = ingest(config)
    assert stats["documents"] > 0
    assert stats["chunks"] > 0

    db_path = config.index_dir / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    cur = storage.conn.execute("SELECT COUNT(*) AS c FROM documents")
    assert cur.fetchone()["c"] > 0
    cur = storage.conn.execute("SELECT COUNT(*) AS c FROM chunks")
    assert cur.fetchone()["c"] > 0
    cur = storage.conn.execute("SELECT COUNT(*) AS c FROM chunks_fts")
    assert cur.fetchone()["c"] > 0
    storage.close()

    chroma_dir = config.index_dir / "chroma"
    assert chroma_dir.exists()
    from chinvex.vectors import VectorStore

    vectors = VectorStore(chroma_dir)
    assert vectors.count() > 0


def test_search_returns_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path)
    chat_dir = _write_chat(tmp_path)
    config = _make_config(tmp_path, repo, chat_dir)

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr("chinvex.search.OllamaEmbedder", FakeEmbedder)

    ingest(config)
    results = search(config, "hello", k=5)
    assert results
