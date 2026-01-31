from fastapi.testclient import TestClient
from fastapi import FastAPI
from chinvex.gateway.endpoints.health import router
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_endpoint_returns_ok():
    """Should return status ok without auth."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "contexts_available" in data


def test_health_endpoint_returns_embedding_config():
    """Should return embedding provider, model, and uptime in health check."""
    # Set up the gateway state directly
    from chinvex.gateway.endpoints.health import set_gateway_state

    set_gateway_state(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        contexts_loaded=2
    )

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Verify embedding config is present
    assert data["embedding_provider"] == "openai"
    assert data["embedding_model"] == "text-embedding-3-small"
    assert data["contexts_loaded"] == 2
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0
