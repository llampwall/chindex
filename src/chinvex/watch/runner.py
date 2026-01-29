import logging
import json
import os
from datetime import datetime
from pathlib import Path
from chinvex.state.models import WatchHit

log = logging.getLogger(__name__)

def run_watches(
    context,
    new_chunk_ids: list[str],
    watches: list,
    timeout_per_watch: int = 30
) -> list[WatchHit]:
    """
    Run all enabled watches against newly ingested chunks.

    Args:
        context: ContextConfig object
        new_chunk_ids: List of newly created chunk IDs
        watches: List of Watch objects
        timeout_per_watch: Timeout in seconds per watch

    Returns:
        List of WatchHit objects for matches

    Note:
        Timeouts and errors are logged but don't fail entire run.
    """
    hits = []

    if not new_chunk_ids:
        return hits

    # Convert new_chunk_ids to set for fast lookup
    new_chunk_id_set = set(new_chunk_ids)

    for watch in watches:
        if not watch.enabled:
            continue

        try:
            # Import search function (avoid circular import)
            from chinvex.search import search_chunks
            from chinvex.storage import Storage
            from chinvex.vectors import VectorStore
            from chinvex.embed import OllamaEmbedder

            # Initialize search components
            db_path = context.index.sqlite_path
            chroma_dir = context.index.chroma_dir
            storage = Storage(db_path)
            vectors = VectorStore(chroma_dir)
            embedder = OllamaEmbedder(context.ollama.base_url, context.ollama.embed_model)

            # Search all chunks
            results = search_chunks(
                storage=storage,
                vectors=vectors,
                embedder=embedder,
                query=watch.query,
                k=50,  # Get more results to filter down
                min_score=0.0,  # Filter by min_score after
                weights=context.weights
            )

            # Filter to only new chunks and apply min_score
            matching = [
                r for r in results
                if r.chunk_id in new_chunk_id_set and r.score >= watch.min_score
            ]

            if matching:
                hits.append(WatchHit(
                    watch_id=watch.id,
                    query=watch.query,
                    hits=[
                        {
                            "chunk_id": r.chunk_id,
                            "score": r.score,
                            "snippet": r.row["text"][:200] if r.row else ""
                        }
                        for r in matching[:5]
                    ],
                    triggered_at=datetime.now()
                ))

        except TimeoutError:
            log.warning(f"Watch {watch.id} timed out after {timeout_per_watch}s, skipping")
            continue
        except Exception as e:
            log.error(f"Watch {watch.id} failed: {e}")
            continue

    return hits


def append_watch_history(history_file: str, entry: dict):
    """
    Append watch history entry to JSONL log.

    Creates file if it doesn't exist.
    """
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def create_watch_history_entry(
    watch_id: str,
    query: str,
    hits: list[dict],
    run_id: str
) -> dict:
    """
    Create watch history entry with hit capping.

    Caps hits at 10 and marks as truncated if exceeded.
    """
    truncated = len(hits) > 10
    capped_hits = hits[:10]

    # Extract snippet (first 200 chars) from each hit
    formatted_hits = []
    for hit in capped_hits:
        formatted_hits.append({
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "snippet": hit.get("snippet", hit.get("text", ""))[:200]
        })

    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "watch_id": watch_id,
        "query": query,
        "hits": formatted_hits,
    }

    if truncated:
        entry["truncated"] = True

    return entry


def should_notify(hits: list, min_score: float) -> bool:
    """Check if any hit meets minimum score threshold."""
    return any(hit.score >= min_score for hit in hits)


def trigger_watch_webhook(config, watch, hits: list) -> bool:
    """
    Trigger webhook notification for watch hit.

    Returns True if webhook sent successfully, False otherwise.
    Does NOT raise exceptions (failures are logged).
    """
    from ..notifications import send_webhook, create_watch_hit_payload

    if not hasattr(config, 'notifications') or config.notifications is None or not config.notifications.enabled:
        return False

    if not config.notifications.webhook_url:
        return False

    # Check min score threshold
    if not should_notify(hits, config.notifications.min_score_for_notify):
        return False

    # Create payload
    hits_data = []
    for h in hits:
        hits_data.append({
            "chunk_id": h.chunk_id,
            "score": h.score,
            "text": h.text if hasattr(h, 'text') else getattr(h, 'snippet', ''),
            "source_uri": h.source_uri if hasattr(h, 'source_uri') else "unknown"
        })

    payload = create_watch_hit_payload(watch.id, watch.query, hits_data)

    # Resolve secret
    secret = config.notifications.webhook_secret
    if secret.startswith("env:"):
        env_var = secret[4:]
        secret = os.environ.get(env_var, "")

    # Send webhook (with retry)
    try:
        return send_webhook(
            config.notifications.webhook_url,
            payload,
            secret=secret if secret else None,
            retry_count=config.notifications.retry_count,
            retry_delay_sec=config.notifications.retry_delay_sec
        )
    except Exception as e:
        print(f"Webhook notification failed: {e}")
        return False
