from __future__ import annotations

import logging
import time
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


def _batched(iterable: list[str], max_items: int, max_bytes: int) -> Iterable[list[str]]:
    """Batch texts by count AND size to avoid massive payloads."""
    batch = []
    size = 0
    for item in iterable:
        item_size = len(item)
        if batch and (len(batch) >= max_items or size + item_size >= max_bytes):
            yield batch
            batch = []
            size = 0
        batch.append(item)
        size += item_size
    if batch:
        yield batch


def _make_session() -> requests.Session:
    """Create session with connection pooling (no auto-retry - we handle that at app level)."""
    s = requests.Session()
    retry = Retry(total=0, backoff_factor=0)
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    return s


class OllamaEmbedder:
    def __init__(
        self,
        host: str,
        model: str,
        fallback_host: str | None = None,
        max_items_per_batch: int = 64,
        max_payload_bytes: int = 1_000_000,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.fallback_host = fallback_host.rstrip("/") if fallback_host else None
        self.max_items_per_batch = max_items_per_batch
        self.max_payload_bytes = max_payload_bytes
        self.session = _make_session()

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
        """Embed texts in safe batches with retry logic."""
        url = f"{self.host}/api/embed"
        out: list[list[float]] = []
        timeout = (5, 180)  # connect timeout 5s, read timeout 180s

        # Guard against chunks larger than max_payload_bytes
        oversized_limit = int(self.max_payload_bytes * 0.8)
        for i, text in enumerate(texts):
            if len(text) > oversized_limit:
                log.warning(
                    f"Chunk {i} size ({len(text)} bytes) exceeds 80% of max_payload_bytes "
                    f"({oversized_limit} bytes). Consider splitting this chunk further."
                )

        batches = list(_batched(texts, self.max_items_per_batch, self.max_payload_bytes))
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches, 1):
            payload = {"model": self.model, "input": batch}
            batch_bytes = sum(len(t) for t in batch)

            # Retry up to 3 times on transient transport errors
            for attempt in range(3):
                try:
                    start_time = time.time()
                    log.debug(
                        f"Embedding batch {batch_idx}/{total_batches}: "
                        f"{len(batch)} items, ~{batch_bytes} bytes, attempt {attempt + 1}/3"
                    )

                    resp = self.session.post(url, json=payload, timeout=timeout)
                    latency_ms = (time.time() - start_time) * 1000

                    # Handle 404 (old Ollama) or context length errors - fallback to single
                    if resp.status_code == 404:
                        out.extend([self._embed_single(t) for t in batch])
                        break
                    if resp.status_code in {400, 500} and self._is_context_length_error(resp):
                        out.extend([self._embed_single(t) for t in batch])
                        break

                    # Other errors
                    if resp.status_code >= 400:
                        self._raise_ollama_error(resp)

                    # Success - parse embeddings
                    data = resp.json()
                    embeddings = data.get("embeddings")
                    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (int, float)):
                        out.append(embeddings)
                    elif not embeddings:
                        raise RuntimeError(f"Ollama response missing embeddings field: {data}")
                    else:
                        out.extend(embeddings)

                    log.debug(
                        f"Batch {batch_idx}/{total_batches} succeeded: "
                        f"{latency_ms:.0f}ms, {len(embeddings)} embeddings"
                    )
                    break  # success

                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ChunkedEncodingError,
                    ConnectionResetError,
                ) as e:
                    if attempt == 2:  # last attempt
                        log.error(
                            f"Batch {batch_idx}/{total_batches} failed after 3 attempts: {type(e).__name__}"
                        )
                        raise
                    # Exponential backoff: 0.5s → 1s → 2s
                    backoff = 0.5 * (2 ** attempt)
                    log.warning(
                        f"Batch {batch_idx}/{total_batches} retry {attempt + 1}/3 "
                        f"after {type(e).__name__}, backoff {backoff}s"
                    )
                    time.sleep(backoff)

        if len(out) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: got {len(out)} expected {len(texts)}")

        return out

    def _embed_single(self, text: str) -> list[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        timeout = (5, 180)  # connect timeout 5s, read timeout 180s
        resp = self.session.post(url, json=payload, timeout=timeout)
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
