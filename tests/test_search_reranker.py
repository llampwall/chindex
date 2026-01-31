from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path
from chinvex.search import search_context
from chinvex.context import ContextConfig


def test_search_with_reranker_enabled():
    """Test that reranker is invoked when configured in context."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = MagicMock(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5,
    )

    # Mock search_chunks to return 15 candidates
    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(15)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            mock_reranker = MagicMock()
            # Reranker returns top 5
            mock_reranker.rerank.return_value = [
                {"chunk_id": f"c{i}", "text": f"text {i}", "rerank_score": 0.95 - i * 0.05}
                for i in range(5)
            ]
            mock_get_reranker.return_value = mock_reranker

            results = search_context(mock_ctx, "test query", k=5, rerank=True)

            # Verify reranker was called
            mock_get_reranker.assert_called_once()
            mock_reranker.rerank.assert_called_once()
            # Verify candidates passed to reranker (15 chunks)
            call_args = mock_reranker.rerank.call_args
            assert call_args[0][0] == "test query"
            assert len(call_args[0][1]) == 15
            # Verify top 5 returned
            assert len(results) == 5


def test_search_with_reranker_disabled():
    """Test that reranker is NOT invoked when not configured."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = None  # No reranker configured

    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(5)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            results = search_context(mock_ctx, "test query", k=5, rerank=False)

            # Verify reranker was NOT called
            mock_get_reranker.assert_not_called()
            assert len(results) == 5


def test_search_reranker_fallback_on_failure():
    """Test that search falls back to pre-rerank results if reranker fails."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = MagicMock(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5,
    )

    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(15)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            mock_reranker = MagicMock()
            mock_reranker.rerank.side_effect = Exception("API timeout")
            mock_get_reranker.return_value = mock_reranker

            with patch("sys.stderr") as mock_stderr:
                results = search_context(mock_ctx, "test query", k=5, rerank=True)

                # Verify warning printed
                assert any("Reranker failed" in str(call) for call in mock_stderr.write.call_args_list)
                # Verify fallback to pre-rerank results (top 5 by original score)
                assert len(results) == 5
                assert results[0].chunk_id == "c0"


def test_search_reranker_skipped_for_few_candidates():
    """Test that reranker is skipped if fewer than 10 candidates."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = MagicMock(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5,
    )

    # Only 8 candidates (below threshold of 10)
    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(8)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            results = search_context(mock_ctx, "test query", k=5, rerank=True)

            # Verify reranker was NOT called (too few candidates)
            mock_get_reranker.assert_not_called()
            assert len(results) == 5
