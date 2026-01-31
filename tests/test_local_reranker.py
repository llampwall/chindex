from unittest.mock import MagicMock, patch
import pytest
from chinvex.rerankers.local import LocalReranker, LocalRerankerConfig


def test_local_reranker_initialization():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    with patch("chinvex.rerankers.local.CrossEncoder") as mock_ce:
        mock_model = MagicMock()
        mock_ce.return_value = mock_model
        reranker = LocalReranker(config)
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.config.top_k == 5
        mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_local_reranker_rerank_success():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    candidates = [
        {"chunk_id": "c1", "text": "Python is a programming language"},
        {"chunk_id": "c2", "text": "Java is also a programming language"},
        {"chunk_id": "c3", "text": "The sky is blue"},
    ]

    mock_model = MagicMock()
    # Mock predict to return scores in descending order
    mock_model.predict.return_value = [0.95, 0.85, 0.25]

    with patch("chinvex.rerankers.local.CrossEncoder", return_value=mock_model):
        reranker = LocalReranker(config)
        results = reranker.rerank("Python programming", candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_id"] == "c2"
    assert results[1]["rerank_score"] == 0.85


def test_local_reranker_truncates_candidates():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    # Create 60 candidates (exceeds MAX_CANDIDATES=50)
    candidates = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(60)]

    mock_model = MagicMock()
    # Return 50 scores (truncated)
    mock_model.predict.return_value = [0.9 - i * 0.01 for i in range(50)]

    with patch("chinvex.rerankers.local.CrossEncoder", return_value=mock_model):
        reranker = LocalReranker(config)
        results = reranker.rerank("test query", candidates)

    # Verify only 50 candidates processed
    assert len(results) == 5
    assert mock_model.predict.call_count == 1
    # Check that predict was called with 50 pairs
    call_args = mock_model.predict.call_args[0][0]
    assert len(call_args) == 50


def test_local_reranker_model_loading_failure():
    config = LocalRerankerConfig(
        provider="local",
        model="invalid-model-name",
        candidates=20,
        top_k=5,
    )

    with patch("chinvex.rerankers.local.CrossEncoder", side_effect=Exception("Model not found")):
        with pytest.raises(Exception, match="Model not found"):
            LocalReranker(config)


def test_local_reranker_empty_candidates():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    candidates = []

    mock_model = MagicMock()
    mock_model.predict.return_value = []

    with patch("chinvex.rerankers.local.CrossEncoder", return_value=mock_model):
        reranker = LocalReranker(config)
        results = reranker.rerank("test query", candidates)

    assert results == []
