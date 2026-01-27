# tests/test_appserver_client.py
from pathlib import Path
import json
import pytest
from chinvex.adapters.cx_appserver.client import AppServerClient
from chinvex.adapters.cx_appserver.capture import capture_raw_response


def test_appserver_client_list_threads(tmp_path: Path, monkeypatch) -> None:
    # Mock HTTP response
    mock_response = {
        "threads": [
            {"id": "thread-1", "title": "Test 1", "created_at": "2026-01-26T10:00:00Z", "updated_at": "2026-01-26T10:30:00Z"},
            {"id": "thread-2", "title": "Test 2", "created_at": "2026-01-25T08:00:00Z", "updated_at": "2026-01-25T08:30:00Z"}
        ]
    }

    def mock_get(url: str, **kwargs):
        class MockResp:
            def json(self):
                return mock_response
            def raise_for_status(self):
                pass
        return MockResp()

    monkeypatch.setattr("requests.get", mock_get)

    client = AppServerClient("http://localhost:8080")
    threads = client.list_threads()

    assert len(threads) == 2
    assert threads[0]["id"] == "thread-1"


def test_appserver_client_get_thread(tmp_path: Path, monkeypatch) -> None:
    mock_response = {
        "id": "thread-123",
        "title": "Test Thread",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:30:00Z",
        "turns": [
            {"turn_id": "turn-1", "ts": "2026-01-26T10:00:00Z", "role": "user", "text": "hello"}
        ]
    }

    def mock_get(url: str, **kwargs):
        class MockResp:
            def json(self):
                return mock_response
            def raise_for_status(self):
                pass
        return MockResp()

    monkeypatch.setattr("requests.get", mock_get)

    client = AppServerClient("http://localhost:8080")
    thread = client.get_thread("thread-123")

    assert thread["id"] == "thread-123"
    assert len(thread["turns"]) == 1


def test_capture_raw_response_writes_file(tmp_path: Path) -> None:
    data = {"key": "value"}
    output_dir = tmp_path / "debug" / "appserver_samples"

    filepath = capture_raw_response(data, "test_endpoint", output_dir)

    assert filepath.exists()
    content = json.loads(filepath.read_text())
    assert content["key"] == "value"
    assert "test_endpoint" in filepath.name
