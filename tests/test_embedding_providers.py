import pytest
from chinvex.embedding_providers import EmbeddingProvider, OllamaProvider, OpenAIProvider

def test_embedding_provider_protocol():
    """Test that providers implement the protocol."""
    def accepts_provider(provider: EmbeddingProvider):
        _ = provider.dimensions
        _ = provider.model_name
        _ = provider.embed(["test"])

    # This should type-check with mypy

def test_ollama_provider_dimensions():
    provider = OllamaProvider("http://localhost:11434", "mxbai-embed-large")
    assert provider.dimensions == 1024
    assert provider.model_name == "mxbai-embed-large"

def test_openai_provider_dimensions():
    provider = OpenAIProvider(api_key="test", model="text-embedding-3-small")
    assert provider.dimensions == 1536
    assert provider.model_name == "text-embedding-3-small"

def test_openai_provider_requires_api_key(monkeypatch):
    # Clear environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key=None, model="text-embedding-3-small")
