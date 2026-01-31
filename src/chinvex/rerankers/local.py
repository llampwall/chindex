from __future__ import annotations

from dataclasses import dataclass
from sentence_transformers import CrossEncoder


@dataclass
class LocalRerankerConfig:
    provider: str
    model: str
    candidates: int
    top_k: int


class LocalReranker:
    """Local cross-encoder reranker using sentence-transformers.

    Downloads model to ~/.cache/huggingface/ on first use.
    Slower than API-based rerankers but free (no API key needed).
    """

    MAX_CANDIDATES = 50

    def __init__(self, config: LocalRerankerConfig):
        self.config = config
        self.model_name = config.model
        # Load cross-encoder model (downloads on first use)
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidates using local cross-encoder model.

        Args:
            query: Search query string
            candidates: List of candidate dicts with 'chunk_id' and 'text' fields
            top_k: Number of results to return (default: use config.top_k)

        Returns:
            Reranked candidates (top K) with 'rerank_score' field added
        """
        if top_k is None:
            top_k = self.config.top_k

        if not candidates:
            return []

        # Budget guardrail: truncate to max 50 candidates
        if len(candidates) > self.MAX_CANDIDATES:
            candidates = candidates[:self.MAX_CANDIDATES]

        # Prepare query-document pairs for cross-encoder
        pairs = [[query, c["text"]] for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores to candidates
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)

        # Sort by score descending and return top K
        scored_candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
