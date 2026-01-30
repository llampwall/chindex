# tests/test_notify.py
import pytest
from unittest.mock import patch, Mock
from pathlib import Path
from chinvex.notify import send_ntfy_push, NtfyConfig


def test_send_ntfy_push_basic():
    """Should send HTTP POST to ntfy server"""
    config = NtfyConfig(
        server="https://ntfy.sh",
        topic="test-topic"
    )

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        send_ntfy_push(config, "Test message")

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        assert "ntfy.sh/test-topic" in call_args[0][0]
        assert call_args[1]["data"] == b"Test message"


def test_send_ntfy_push_with_title():
    """Should include title in headers if provided"""
    config = NtfyConfig(server="https://ntfy.sh", topic="test")

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        send_ntfy_push(config, "Body", title="Test Title")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Title"] == "Test Title"


def test_send_ntfy_push_with_priority():
    """Should include priority in headers"""
    config = NtfyConfig(server="https://ntfy.sh", topic="test")

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        send_ntfy_push(config, "Urgent!", priority="high")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Priority"] == "high"


def test_send_ntfy_push_disabled():
    """Should skip if enabled=False"""
    config = NtfyConfig(
        server="https://ntfy.sh",
        topic="test",
        enabled=False
    )

    with patch('requests.post') as mock_post:
        send_ntfy_push(config, "Test")

        # Should not have called POST
        mock_post.assert_not_called()


def test_should_send_stale_alert_dedup(tmp_path: Path):
    """Should only send stale alert once per context per day"""
    from chinvex.notify import should_send_stale_alert, log_push

    log_file = tmp_path / "push_log.jsonl"

    # First call - should allow
    assert should_send_stale_alert("TestCtx", log_file) is True

    # Log the push
    log_push("TestCtx", "stale", log_file)

    # Second call same day - should block
    assert should_send_stale_alert("TestCtx", log_file) is False

    # Different context - should allow
    assert should_send_stale_alert("OtherCtx", log_file) is True


def test_log_push_records_to_file(tmp_path: Path):
    """Should append push records to JSONL log"""
    from chinvex.notify import log_push

    log_file = tmp_path / "push_log.jsonl"

    log_push("TestCtx", "stale", log_file)
    log_push("TestCtx", "watch_hit", log_file)

    # Should have 2 lines
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    # Should be valid JSON
    import json
    record1 = json.loads(lines[0])
    assert record1["context"] == "TestCtx"
    assert record1["type"] == "stale"
    assert "timestamp" in record1
    assert "date" in record1
