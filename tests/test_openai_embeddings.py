import pytest
from unittest.mock import Mock, patch
from chinvex.embedding_providers import OpenAIProvider

def test_openai_embed_calls_api():
    """Test OpenAI embedding makes API call with correct params."""
    with patch("chinvex.embedding_providers.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3] * 512)  # 1536 dims
        ]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")
        result = provider.embed(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_client.embeddings.create.assert_called_once()

def test_openai_embed_batching():
    """Test OpenAI respects batch size limits."""
    texts = ["text"] * 3000  # Exceeds batch limit of 2048

    with patch("chinvex.embedding_providers.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create responses for each batch
        mock_response_1 = Mock()
        mock_response_1.data = [Mock(embedding=[0.1] * 1536) for _ in range(2048)]
        mock_response_2 = Mock()
        mock_response_2.data = [Mock(embedding=[0.2] * 1536) for _ in range(952)]

        mock_client.embeddings.create.side_effect = [mock_response_1, mock_response_2]

        provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")
        result = provider.embed(texts)

        # Should make 2 API calls (2048 + 952)
        assert mock_client.embeddings.create.call_count == 2
        assert len(result) == 3000

def test_openai_embed_retry_on_rate_limit():
    """Test OpenAI retries on rate limit errors."""
    with patch("chinvex.embedding_providers.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # First call raises rate limit, second succeeds
        from openai import RateLimitError
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]

        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(status_code=429), body=None),
            mock_response
        ]

        provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")
        result = provider.embed(["test"])

        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 2
