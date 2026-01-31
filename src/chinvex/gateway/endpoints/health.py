"""Health check endpoint."""

from fastapi import APIRouter
from datetime import datetime, timezone
from chinvex.gateway import __version__
from chinvex.context import list_contexts
from chinvex.context_cli import get_contexts_root


router = APIRouter()

# Module-level state set during startup
_gateway_state = {
    "embedding_provider": None,
    "embedding_model": None,
    "contexts_loaded": 0,
    "startup_time": None
}


def set_gateway_state(embedding_provider: str, embedding_model: str, contexts_loaded: int):
    """Set gateway state during startup. Called from app.py startup event."""
    _gateway_state["embedding_provider"] = embedding_provider
    _gateway_state["embedding_model"] = embedding_model
    _gateway_state["contexts_loaded"] = contexts_loaded
    _gateway_state["startup_time"] = datetime.now(timezone.utc)


def get_gateway_state() -> dict:
    """Get current gateway state for testing."""
    return _gateway_state.copy()


@router.get("/health")
async def health():
    """
    Health check endpoint. No authentication required.

    Returns:
        Status information including version, context count, and embedding configuration
    """
    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        contexts_available = len(contexts)
    except Exception:
        # If context listing fails, don't fail health check
        contexts_available = 0

    # Calculate uptime
    uptime_seconds = 0
    if _gateway_state["startup_time"]:
        uptime_seconds = int((datetime.now(timezone.utc) - _gateway_state["startup_time"]).total_seconds())

    response = {
        "status": "ok",
        "version": __version__,
        "contexts_available": contexts_available
    }

    # Include embedding config if available
    if _gateway_state["embedding_provider"]:
        response["embedding_provider"] = _gateway_state["embedding_provider"]
        response["embedding_model"] = _gateway_state["embedding_model"]
        response["contexts_loaded"] = _gateway_state["contexts_loaded"]
        response["uptime_seconds"] = uptime_seconds

    return response
