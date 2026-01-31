# tests/test_jina_reranker.py
from unittest.mock import MagicMock, patch
import pytest
from chinvex.rerankers.jina import JinaReranker, JinaRerankerConfig


def test_jina_reranker_initialization():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    with patch("chinvex.rerankers.jina.requests"):
        reranker = JinaReranker(config, api_key="test-key")
        assert reranker.model == "jina-reranker-v1-base-en"
        assert reranker.api_key == "test-key"
        assert reranker.config.top_k == 5


def test_jina_reranker_missing_api_key():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        JinaReranker(config, api_key=None)


def test_jina_reranker_rerank_success():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    candidates = [
        {"chunk_id": "c1", "text": "Python is a programming language"},
        {"chunk_id": "c2", "text": "Java is also a programming language"},
        {"chunk_id": "c3", "text": "The sky is blue"},
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.85},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("chinvex.rerankers.jina.requests.post", return_value=mock_response):
        reranker = JinaReranker(config, api_key="test-key")
        results = reranker.rerank("Python programming", candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_id"] == "c2"
    assert results[1]["rerank_score"] == 0.85


def test_jina_reranker_truncates_candidates():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    # Create 60 candidates (exceeds MAX_CANDIDATES=50)
    candidates = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(60)]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"index": i, "relevance_score": 0.9 - i * 0.01} for i in range(5)]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("chinvex.rerankers.jina.requests.post", return_value=mock_response) as mock_post:
        reranker = JinaReranker(config, api_key="test-key")
        results = reranker.rerank("test query", candidates)

    # Verify only 50 candidates sent to API
    call_args = mock_post.call_args
    sent_documents = call_args[1]["json"]["documents"]
    assert len(sent_documents) == 50
    assert len(results) == 5


def test_jina_reranker_api_failure():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    candidates = [{"chunk_id": "c1", "text": "test"}]

    with patch("chinvex.rerankers.jina.requests.post", side_effect=Exception("API error")):
        reranker = JinaReranker(config, api_key="test-key")
        with pytest.raises(Exception, match="API error"):
            reranker.rerank("query", candidates)
