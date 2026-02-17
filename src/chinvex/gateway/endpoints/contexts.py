"""Contexts endpoint - list available contexts."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from chinvex.context import list_contexts
from chinvex.context_cli import get_contexts_root
from chinvex.gateway.config import load_gateway_config

log = logging.getLogger(__name__)

router = APIRouter()


class ContextInfo(BaseModel):
    """Context information."""
    name: str
    aliases: list[str]
    updated_at: str
    status: str | None = None
    file_count: int | None = None
    chunk_count: int | None = None


class ContextsResponse(BaseModel):
    """Response from contexts endpoint."""
    contexts: list[ContextInfo]


def _read_status(contexts_root: Path, name: str) -> dict:
    """Read STATUS.json for a context, returning empty dict on failure."""
    status_file = contexts_root / name / "STATUS.json"
    if not status_file.exists():
        return {}
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _derive_status(status_data: dict) -> str:
    """Derive context status from STATUS.json data."""
    freshness = status_data.get("freshness", {})
    if freshness.get("is_stale", False):
        return "stale"
    return "synced"


@router.get("/contexts", response_model=ContextsResponse)
async def list_available_contexts():
    """
    List available contexts. Respects allowlist.
    Returns file/chunk counts and sync status from STATUS.json.
    """
    contexts_root = get_contexts_root()
    all_contexts = list_contexts(contexts_root)
    config = load_gateway_config()

    # Filter by allowlist if configured
    if config.context_allowlist:
        filtered = [c for c in all_contexts if c.name in config.context_allowlist]
    else:
        filtered = all_contexts

    result = []
    for c in filtered:
        status_data = _read_status(contexts_root, c.name)
        result.append(ContextInfo(
            name=c.name,
            aliases=c.aliases,
            updated_at=status_data.get("last_sync", c.updated_at),
            status=_derive_status(status_data) if status_data else None,
            file_count=status_data.get("documents", status_data.get("files", None)),
            chunk_count=status_data.get("chunks", None),
        ))

    return ContextsResponse(contexts=result)
