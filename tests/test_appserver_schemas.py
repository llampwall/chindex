# tests/test_appserver_schemas.py
import pytest
from pydantic import ValidationError
from chinvex.adapters.cx_appserver.schemas import AppServerThread, AppServerTurn


def test_appserver_thread_schema_valid() -> None:
    data = {
        "id": "thread-123",
        "title": "Test Thread",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:30:00Z",
        "turns": []
    }

    thread = AppServerThread.model_validate(data)
    assert thread.id == "thread-123"
    assert thread.title == "Test Thread"


def test_appserver_thread_schema_missing_required_fails() -> None:
    data = {"title": "Missing ID"}

    with pytest.raises(ValidationError):
        AppServerThread.model_validate(data)


def test_appserver_turn_schema_valid() -> None:
    data = {
        "turn_id": "turn-1",
        "ts": "2026-01-26T10:00:00Z",
        "role": "user",
        "text": "hello"
    }

    turn = AppServerTurn.model_validate(data)
    assert turn.turn_id == "turn-1"
    assert turn.role == "user"
    assert turn.text == "hello"


def test_appserver_turn_with_tool() -> None:
    data = {
        "turn_id": "turn-tool",
        "ts": "2026-01-26T10:00:00Z",
        "role": "tool",
        "text": "",
        "tool": {
            "name": "bash",
            "input": {"command": "ls"},
            "output": {"stdout": "file1.txt"}
        }
    }

    turn = AppServerTurn.model_validate(data)
    assert turn.tool is not None
    assert turn.tool["name"] == "bash"
