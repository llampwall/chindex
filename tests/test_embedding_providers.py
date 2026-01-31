"""Tests for embedding provider selection and defaults."""

import pytest
import os
from unittest.mock import patch
from chinvex.embedding_providers import get_provider, OpenAIProvider, OllamaProvider


def test_get_provider_defaults_to_openai():
    """When no provider specified, should default to OpenAI (P5 spec)."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        provider = get_provider(
            cli_provider=None,
            context_config=None,
            env_provider=None
        )

    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimensions == 1536


def test_get_provider_cli_overrides_all():
    """CLI flag should override context.json and env."""
    context_config = {
        "embedding": {
            "provider": "ollama",
            "model": "mxbai-embed-large"
        }
    }

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "CHINVEX_EMBED_PROVIDER": "ollama"}):
        provider = get_provider(
            cli_provider="openai",
            context_config=context_config,
            env_provider="ollama"
        )

    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "text-embedding-3-small"


def test_get_provider_context_config_overrides_env():
    """context.json should override environment variable."""
    context_config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        }
    }

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "CHINVEX_EMBED_PROVIDER": "ollama"}):
        provider = get_provider(
            cli_provider=None,
            context_config=context_config,
            env_provider="ollama"
        )

    assert isinstance(provider, OpenAIProvider)


def test_get_provider_env_overrides_default():
    """Environment variable should override default."""
    with patch.dict(os.environ, {"CHINVEX_EMBED_PROVIDER": "ollama"}):
        provider = get_provider(
            cli_provider=None,
            context_config=None,
            env_provider="ollama"
        )

    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "mxbai-embed-large"


def test_get_provider_explicit_ollama():
    """Should still support explicit Ollama selection."""
    provider = get_provider(
        cli_provider="ollama",
        context_config=None,
        env_provider=None
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "mxbai-embed-large"
    assert provider.dimensions == 1024


def test_openai_provider_requires_api_key():
    """OpenAI provider should fail if API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(api_key=None, model="text-embedding-3-small")


def test_openai_provider_with_explicit_key():
    """OpenAI provider should accept explicit API key."""
    provider = OpenAIProvider(api_key="sk-test-key", model="text-embedding-3-small")

    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimensions == 1536


# Legacy tests (kept for backward compatibility)
def test_embedding_provider_protocol():
    """Test that providers implement the protocol."""
    def accepts_provider(provider):
        _ = provider.dimensions
        _ = provider.model_name
        _ = provider.embed(["test"])


def test_ollama_provider_dimensions():
    provider = OllamaProvider("http://localhost:11434", "mxbai-embed-large")
    assert provider.dimensions == 1024
    assert provider.model_name == "mxbai-embed-large"


def test_openai_provider_dimensions():
    provider = OpenAIProvider(api_key="test", model="text-embedding-3-small")
    assert provider.dimensions == 1536
    assert provider.model_name == "text-embedding-3-small"
