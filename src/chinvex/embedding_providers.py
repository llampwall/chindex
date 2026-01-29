from __future__ import annotations

import logging
import os
from typing import Protocol

log = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class OllamaProvider:
    """Ollama embedding provider."""

    # Model dimensions (hardcoded for now, could query Ollama API)
    MODEL_DIMS = {
        "mxbai-embed-large": 1024,
    }

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown Ollama model: {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Implementation will use existing OllamaEmbedder
        from .embed import OllamaEmbedder
        embedder = OllamaEmbedder(self.host, self.model)
        return embedder.embed(texts)

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS[self.model]

    @property
    def model_name(self) -> str:
        return self.model


class OpenAIProvider:
    """OpenAI embedding provider."""

    # Model dimensions
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown OpenAI model: {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Will implement OpenAI API call in next task
        raise NotImplementedError("OpenAI embedding not yet implemented")

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS[self.model]

    @property
    def model_name(self) -> str:
        return self.model
