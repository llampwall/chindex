# src/chinvex/rerankers/cohere.py
from __future__ import annotations

import os
from typing import Any

try:
    import cohere
except ImportError:
    cohere = None

from ..reranker_config import RerankerConfig


class CohereReranker:
    """Cohere Rerank API provider for two-stage retrieval.

    Requires COHERE_API_KEY environment variable.
    """

    MAX_CANDIDATES = 50  # Budget guardrail

    def __init__(self, config: RerankerConfig):
        if cohere is None:
            raise ImportError(
                "cohere package not installed. Install with: pip install cohere"
            )

        self.config = config
        self.model = config.model

        # Get API key from environment
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY environment variable required for Cohere reranker"
            )

        self.client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using Cohere Rerank API.

        Args:
            query: Search query string
            candidates: List of candidate dicts with 'chunk_id' and 'text' fields
            top_k: Number of results to return (default: use config.top_k)

        Returns:
            Reranked candidates (top K) with 'rerank_score' field added

        Raises:
            Exception: If API call fails or times out
        """
        if top_k is None:
            top_k = self.config.top_k

        # Budget guardrail: truncate to max 50 candidates
        if len(candidates) > self.MAX_CANDIDATES:
            candidates = candidates[:self.MAX_CANDIDATES]

        # Extract text for reranking
        documents = [c["text"] for c in candidates]

        # Call Cohere Rerank API
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k,
        )

        # Map results back to original candidates
        reranked = []
        for result in response.results:
            candidate = candidates[result.index].copy()
            candidate["rerank_score"] = result.relevance_score
            reranked.append(candidate)

        return reranked
