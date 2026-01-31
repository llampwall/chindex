"""Chinvex MCP Server - HTTP client for Chinvex gateway.

Exposes Chinvex memory search to Claude Desktop/Code via MCP protocol.
Configured via environment variables:
  - CHINVEX_URL: Gateway URL (default: https://chinvex.yourdomain.com)
  - CHINVEX_API_TOKEN: Bearer token for authentication
"""

from __future__ import annotations

import os
import json
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# Configuration from environment
CHINVEX_URL = os.environ.get("CHINVEX_URL", "https://chinvex.yourdomain.com")
CHINVEX_API_TOKEN = os.environ.get("CHINVEX_API_TOKEN", "")

if not CHINVEX_API_TOKEN:
    raise RuntimeError("CHINVEX_API_TOKEN environment variable is required")

# Initialize MCP server
mcp = FastMCP("chinvex_mcp")

# HTTP client
_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    """Get or create HTTP client with auth headers."""
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=CHINVEX_URL,
            headers={"Authorization": f"Bearer {CHINVEX_API_TOKEN}"},
            timeout=30.0,
        )
    return _client


# --- Input Models ---


class SearchInput(BaseModel):
    """Input for chinvex_search tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Search query text", min_length=1)
    contexts: Optional[str] = Field(
        default="all",
        description="Context(s) to search. Use 'all' for cross-context, or specific name like 'Chinvex'",
    )
    k: int = Field(default=8, description="Number of results to return", ge=1, le=50)


class ListContextsInput(BaseModel):
    """Input for chinvex_list_contexts tool (no parameters needed)."""

    model_config = ConfigDict(extra="forbid")


class GetChunksInput(BaseModel):
    """Input for chinvex_get_chunks tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    context: str = Field(..., description="Context name", min_length=1)
    chunk_ids: list[str] = Field(
        ..., description="List of chunk IDs to retrieve", min_length=1
    )


# --- Tools ---


@mcp.tool(
    name="chinvex_search",
    annotations={
        "title": "Search Chinvex Memory",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def chinvex_search(params: SearchInput) -> str:
    """Search personal knowledge base with grounded retrieval.

    Returns evidence chunks with citations from indexed sources including
    code repos, chat exports, and notes. Use this to find relevant context
    about past conversations, project details, or documented information.

    Args:
        params: Search parameters including query, contexts, and result count

    Returns:
        JSON with grounded status and matching chunks with source citations
    """
    client = get_client()

    # Build request - use contexts field for cross-context search
    payload = {
        "query": params.query,
        "k": params.k,
    }
    if params.contexts == "all":
        payload["contexts"] = "all"
    else:
        payload["context"] = params.contexts  # singular field for single context

    try:
        resp = client.post("/v1/evidence", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Format response for readability
        # Extract chunks from evidence_pack
        chunks = data.get("evidence_pack", {}).get("chunks", [])

        result = {
            "grounded": data.get("grounded", False),
            "contexts_searched": data.get("contexts_searched", []),
            "query": data.get("query", params.query),
            "total_results": len(chunks),
            "chunks": [],
        }

        for chunk in chunks:
            result["chunks"].append(
                {
                    "id": chunk.get("chunk_id"),
                    "source": chunk.get("source_uri", ""),
                    "context": chunk.get("context", ""),
                    "score": chunk.get("score"),
                    "text": chunk.get("text", "")[:2000],  # Truncate very long chunks
                }
            )

        return json.dumps(result, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps(
            {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Request failed: {type(e).__name__}: {str(e)}"})


@mcp.tool(
    name="chinvex_list_contexts",
    annotations={
        "title": "List Available Contexts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def chinvex_list_contexts(params: ListContextsInput) -> str:
    """List all available contexts in the Chinvex knowledge base.

    Use this to discover what contexts are available for searching.
    Each context represents a project or knowledge domain.

    Returns:
        JSON list of contexts with names, aliases, and last update times
    """
    client = get_client()

    try:
        resp = client.get("/v1/contexts")
        resp.raise_for_status()
        data = resp.json()

        contexts = []
        for ctx in data.get("contexts", []):
            contexts.append(
                {
                    "name": ctx.get("name"),
                    "aliases": ctx.get("aliases", []),
                    "updated_at": ctx.get("updated_at"),
                }
            )

        return json.dumps({"contexts": contexts}, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps(
            {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Request failed: {type(e).__name__}: {str(e)}"})


@mcp.tool(
    name="chinvex_get_chunks",
    annotations={
        "title": "Get Chunks by ID",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def chinvex_get_chunks(params: GetChunksInput) -> str:
    """Retrieve full chunk content by ID.

    Use this after searching to get the complete text of specific chunks
    when you need more context than the search snippet provides.

    Args:
        params: Context name and list of chunk IDs to retrieve

    Returns:
        JSON with full chunk content and metadata
    """
    client = get_client()

    payload = {
        "context": params.context,
        "chunk_ids": params.chunk_ids,
    }

    try:
        resp = client.post("/v1/chunks", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return json.dumps(data, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps(
            {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Request failed: {type(e).__name__}: {str(e)}"})


def main() -> None:
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
