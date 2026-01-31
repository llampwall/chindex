# tests/test_cohere_reranker.py
from unittest.mock import patch, MagicMock

import pytest

from chinvex.rerankers.cohere import CohereReranker
from chinvex.reranker_config import RerankerConfig


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_initialization(mock_cohere_module):
    """CohereReranker should initialize with config."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5
    )

    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        assert reranker.config == config
        assert reranker.model == "rerank-english-v3.0"


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_requires_api_key(mock_cohere_module):
    """CohereReranker should require COHERE_API_KEY environment variable."""
    config = RerankerConfig(provider="cohere", model="rerank-english-v3.0")

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            CohereReranker(config)


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_calls_api(mock_cohere_module):
    """CohereReranker should call Cohere rerank API with correct parameters."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5
    )

    # Mock Cohere client
    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client

    # Mock rerank response
    mock_result = MagicMock()
    mock_result.results = [
        MagicMock(index=2, relevance_score=0.95),
        MagicMock(index=0, relevance_score=0.87),
        MagicMock(index=1, relevance_score=0.72),
    ]
    mock_client.rerank.return_value = mock_result

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        candidates = [
            {"chunk_id": "c1", "text": "First chunk"},
            {"chunk_id": "c2", "text": "Second chunk"},
            {"chunk_id": "c3", "text": "Third chunk"},
        ]

        reranked = reranker.rerank(query="test query", candidates=candidates, top_k=3)

        # Verify API call
        mock_client.rerank.assert_called_once()
        call_kwargs = mock_client.rerank.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["documents"] == ["First chunk", "Second chunk", "Third chunk"]
        assert call_kwargs["model"] == "rerank-english-v3.0"
        assert call_kwargs["top_n"] == 3

        # Verify reranked order (by index in mock results)
        assert len(reranked) == 3
        assert reranked[0]["chunk_id"] == "c3"  # index=2
        assert reranked[0]["rerank_score"] == 0.95
        assert reranked[1]["chunk_id"] == "c1"  # index=0
        assert reranked[1]["rerank_score"] == 0.87
        assert reranked[2]["chunk_id"] == "c2"  # index=1
        assert reranked[2]["rerank_score"] == 0.72


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_handles_timeout(mock_cohere_module):
    """CohereReranker should raise TimeoutError on slow API response."""
    config = RerankerConfig(provider="cohere", model="rerank-english-v3.0")

    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client
    mock_client.rerank.side_effect = Exception("Timeout")

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        candidates = [{"chunk_id": "c1", "text": "test"}]

        with pytest.raises(Exception, match="Timeout"):
            reranker.rerank(query="test", candidates=candidates, top_k=1)


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_truncates_to_max_candidates(mock_cohere_module):
    """CohereReranker should truncate to 50 candidates max (budget guardrail)."""
    config = RerankerConfig(provider="cohere", model="rerank-english-v3.0")

    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client
    mock_client.rerank.return_value = MagicMock(results=[])

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        # Create 60 candidates
        candidates = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(60)]

        reranker.rerank(query="test", candidates=candidates, top_k=5)

        # Should only send 50 to API
        call_kwargs = mock_client.rerank.call_args[1]
        assert len(call_kwargs["documents"]) == 50
