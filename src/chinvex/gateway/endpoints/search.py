"""Search endpoint - raw hybrid search with multi-context support."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chinvex.context import load_context, ContextNotFoundError, list_contexts
from chinvex.context_cli import get_contexts_root
from chinvex.search import hybrid_search_from_context, search_multi_context, search_context
from chinvex.gateway.validation import MultiContextSearchRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    context: str | None = None
    contexts_searched: list[str] | str | None = None
    query: str
    results: list[dict]
    total_results: int


@router.post("/search", response_model=SearchResponse)
async def search(req: MultiContextSearchRequest, request: Request):
    """
    Raw hybrid search. Returns ranked chunks without grounding check.
    Supports single context or multi-context search.
    """
    contexts_root = get_contexts_root()
    config = load_gateway_config()

    # Determine if this is multi-context or single-context search
    if req.contexts is not None:
        # Multi-context search
        if isinstance(req.contexts, str) and req.contexts == "all":
            ctx_list = "all"
            # Apply allowlist filter if configured
            if config.context_allowlist:
                all_ctx = [c.name for c in list_contexts(contexts_root)]
                ctx_list = [c for c in all_ctx if c in config.context_allowlist]
        else:
            ctx_list = req.contexts
            # Apply allowlist filter if configured
            if config.context_allowlist:
                ctx_list = [c for c in ctx_list if c in config.context_allowlist]
                if not ctx_list:
                    raise HTTPException(status_code=404, detail="No allowed contexts found")

        # Map source_types to source parameter
        source = "all"
        if req.source_types:
            if len(req.source_types) == 1:
                source = req.source_types[0]
            # If multiple, use "all" (search_multi_context will handle filtering)

        results = search_multi_context(
            contexts=ctx_list,
            query=req.query,
            k=req.k,
            source=source,
            recency_enabled=not req.no_recency,
        )

        # Convert SearchResult to dict format expected by gateway
        return SearchResponse(
            contexts_searched=ctx_list,
            query=req.query,
            results=[
                {
                    "chunk_id": r.chunk_id,
                    "context": r.context,
                    "text": r.snippet,  # Use snippet for consistency
                    "title": r.title,
                    "citation": r.citation,
                    "source_type": r.source_type,
                    "score": r.score,
                }
                for r in results
            ],
            total_results=len(results)
        )

    elif req.context is not None:
        # Single-context search (backward compatible)
        request.state.context = req.context

        try:
            context = load_context(req.context, contexts_root)
        except ContextNotFoundError:
            raise HTTPException(status_code=404, detail="Context not found")

        if config.context_allowlist and req.context not in config.context_allowlist:
            raise HTTPException(status_code=404, detail="Context not found")

        results = hybrid_search_from_context(
            context=context,
            query=req.query,
            k=req.k,
            source_types=req.source_types,
            no_recency=req.no_recency
        )

        return SearchResponse(
            context=req.context,
            query=req.query,
            results=[
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text[:5000] + (" [truncated]" if len(r.text) > 5000 else ""),
                    "source_uri": r.source_uri,
                    "source_type": r.source_type,
                    "scores": {
                        "fts": r.fts_score,
                        "vector": r.vector_score,
                        "blended": r.blended_score,
                        "rank": r.rank_score
                    },
                    "metadata": {
                        "line_start": getattr(r, 'line_start', None),
                        "line_end": getattr(r, 'line_end', None),
                        "updated_at": getattr(r, 'updated_at', None)
                    }
                }
                for r in results
            ],
            total_results=len(results)
        )
    else:
        raise HTTPException(status_code=400, detail="Must specify either context or contexts")
