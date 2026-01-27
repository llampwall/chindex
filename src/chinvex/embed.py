from __future__ import annotations

import requests


class OllamaEmbedder:
    def __init__(self, host: str, model: str, fallback_host: str | None = None) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.fallback_host = fallback_host.rstrip("/") if fallback_host else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embed_batch(texts)
        except requests.RequestException as exc:
            # Try fallback host if configured
            if self.fallback_host and self.fallback_host != self.host:
                print(f"Warning: {self.host} unreachable, falling back to {self.fallback_host}")
                original_host = self.host
                self.host = self.fallback_host
                try:
                    return self._embed_batch(texts)
                except requests.RequestException:
                    self.host = original_host  # Restore original
                    raise RuntimeError(
                        f"Ollama connection failed on both {original_host} and {self.fallback_host}. "
                        "Ensure Ollama is running and reachable."
                    ) from exc
            raise RuntimeError(
                f"Ollama connection failed on {self.host}. Ensure Ollama is running and reachable."
            ) from exc

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.host}/api/embed"
        payload = {"model": self.model, "input": texts}
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 404:
            return [self._embed_single(t) for t in texts]
        if resp.status_code in {400, 500} and self._is_context_length_error(resp):
            return [self._embed_single(t) for t in texts]
        if resp.status_code >= 400:
            self._raise_ollama_error(resp)
        data = resp.json()
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (int, float)):
            return [embeddings]
        if not embeddings:
            raise RuntimeError(f"Ollama response missing embeddings field: {data}")
        return embeddings

    def _embed_single(self, text: str) -> list[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code in {400, 500} and self._is_context_length_error(resp):
            return self._embed_split(text)
        if resp.status_code >= 400:
            self._raise_ollama_error(resp)
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(f"Ollama response missing embedding field: {data}")
        return embedding

    def _raise_ollama_error(self, resp: requests.Response) -> None:
        body = resp.text.strip()
        raise RuntimeError(f"Ollama request failed ({resp.status_code}). URL={resp.url} Body={body}")

    def _is_context_length_error(self, resp: requests.Response) -> bool:
        try:
            body = resp.text.lower()
        except Exception:
            return False
        return "input length exceeds the context length" in body

    def _embed_split(self, text: str) -> list[float]:
        if len(text) < 2:
            raise RuntimeError("Ollama request failed (400). Input too short to split.")
        left, right = self._split_text(text)
        if not left or not right:
            raise RuntimeError("Ollama request failed (400). Unable to split input safely.")
        emb_left = self._embed_single(left)
        emb_right = self._embed_single(right)
        return self._average_vectors(emb_left, emb_right)

    def _split_text(self, text: str) -> tuple[str, str]:
        mid = len(text) // 2
        max_search = min(500, mid)
        split_at = None
        for offset in range(max_search + 1):
            left_idx = mid - offset
            right_idx = mid + offset
            if left_idx > 0 and text[left_idx].isspace():
                split_at = left_idx
                break
            if right_idx < len(text) and text[right_idx].isspace():
                split_at = right_idx
                break
        if split_at is None:
            split_at = mid
        return text[:split_at].strip(), text[split_at:].strip()

    def _average_vectors(self, a: list[float], b: list[float]) -> list[float]:
        if len(a) != len(b):
            raise RuntimeError("Ollama returned embeddings with different lengths.")
        return [(x + y) / 2.0 for x, y in zip(a, b)]
