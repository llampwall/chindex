# src/chinvex/rerankers/jina.py
from __future__ import annotations

import os
import requests
from dataclasses import dataclass


@dataclass
class JinaRerankerConfig:
    provider: str
    model: str
    candidates: int
    top_k: int


class JinaReranker:
    """Reranker using Jina Reranker API."""

    MAX_CANDIDATES = 50
    API_URL = "https://api.jina.ai/v1/rerank"

    def __init__(self, config: JinaRerankerConfig, api_key: str | None = None):
        self.config = config
        self.model = config.model
        self.api_key = api_key or os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY environment variable required for Jina reranker. "
                "Get your API key from https://jina.ai/"
            )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidates using Jina Rerank API.

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

        # Call Jina Rerank API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=2.0,
        )
        response.raise_for_status()

        data = response.json()

        # Map results back to original candidates
        reranked = []
        for result in data["results"]:
            candidate = candidates[result["index"]].copy()
            candidate["rerank_score"] = result["relevance_score"]
            reranked.append(candidate)

        return reranked
