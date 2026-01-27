# tests/test_appserver_normalize.py
from chinvex.adapters.cx_appserver.normalize import normalize_to_conversation_doc
from chinvex.adapters.cx_appserver.schemas import AppServerThread, AppServerTurn


def test_normalize_appserver_thread_to_conversation_doc() -> None:
    thread = AppServerThread(
        id="thread-abc",
        title="Test Conversation",
        created_at="2026-01-26T10:00:00Z",
        updated_at="2026-01-26T10:30:00Z",
        turns=[
            AppServerTurn(
                turn_id="turn-1",
                ts="2026-01-26T10:00:00Z",
                role="user",
                text="hello"
            ),
            AppServerTurn(
                turn_id="turn-2",
                ts="2026-01-26T10:01:00Z",
                role="assistant",
                text="hi there"
            )
        ],
        links={"workspace_id": "ws-123"}
    )

    doc = normalize_to_conversation_doc(thread)

    assert doc["doc_type"] == "conversation"
    assert doc["source"] == "cx_appserver"
    assert doc["thread_id"] == "thread-abc"
    assert doc["title"] == "Test Conversation"
    assert len(doc["turns"]) == 2
    assert doc["turns"][0]["role"] == "user"
    assert doc["links"]["workspace_id"] == "ws-123"


def test_normalize_preserves_tool_info() -> None:
    thread = AppServerThread(
        id="thread-tool",
        title=None,
        created_at="2026-01-26T10:00:00Z",
        updated_at="2026-01-26T10:00:00Z",
        turns=[
            AppServerTurn(
                turn_id="turn-tool",
                ts="2026-01-26T10:00:00Z",
                role="tool",
                text="",
                tool={"name": "bash", "input": {}, "output": {}}
            )
        ]
    )

    doc = normalize_to_conversation_doc(thread)

    assert doc["turns"][0]["tool"] is not None
    assert doc["turns"][0]["tool"]["name"] == "bash"


def test_normalize_generates_title_from_first_user_message() -> None:
    thread = AppServerThread(
        id="thread-no-title",
        title=None,
        created_at="2026-01-26T10:00:00Z",
        updated_at="2026-01-26T10:00:00Z",
        turns=[
            AppServerTurn(
                turn_id="turn-1",
                ts="2026-01-26T10:00:00Z",
                role="user",
                text="This is a long user message that should be truncated to form the title"
            )
        ]
    )

    doc = normalize_to_conversation_doc(thread)

    assert doc["title"] is not None
    assert len(doc["title"]) <= 60
    assert "This is a long user message" in doc["title"]
