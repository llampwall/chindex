"""Test webhook integration with watch runner."""
import pytest
from unittest.mock import patch, Mock


@patch('chinvex.notifications.send_webhook')
def test_watch_triggers_webhook(mock_send_webhook):
    """Test that watch hit triggers webhook."""
    from chinvex.watch.runner import trigger_watch_webhook

    mock_send_webhook.return_value = True

    # Mock watch and hits
    watch = Mock(id="test_watch", query="test")
    hits = [
        Mock(chunk_id="abc", score=0.85, text="test text", source_uri="file.txt")
    ]

    # Mock config
    config = Mock()
    config.notifications.enabled = True
    config.notifications.webhook_url = "https://example.com/webhook"
    config.notifications.webhook_secret = "secret"
    config.notifications.min_score_for_notify = 0.75
    config.notifications.retry_count = 2
    config.notifications.retry_delay_sec = 5

    # Trigger webhook
    success = trigger_watch_webhook(config, watch, hits)

    assert success is True
    mock_send_webhook.assert_called_once()


def test_watch_webhook_respects_min_score():
    """Test that low-scoring hits don't trigger webhook."""
    from chinvex.watch.runner import should_notify

    # Low score - should not notify
    assert should_notify([Mock(score=0.5)], min_score=0.75) is False

    # High score - should notify
    assert should_notify([Mock(score=0.85)], min_score=0.75) is True


@patch('chinvex.notifications.send_webhook')
def test_watch_webhook_failure_does_not_block(mock_send_webhook):
    """Test that webhook failure doesn't block ingest."""
    from chinvex.watch.runner import trigger_watch_webhook

    mock_send_webhook.return_value = False  # Webhook fails

    # Should return False but not raise exception
    watch = Mock(id="test_watch", query="test")
    hits = [Mock(chunk_id="abc", score=0.85, text="test text", source_uri="file.txt")]
    config = Mock()
    config.notifications.enabled = True
    config.notifications.webhook_url = "https://example.com/webhook"
    config.notifications.webhook_secret = ""
    config.notifications.min_score_for_notify = 0.75
    config.notifications.retry_count = 2
    config.notifications.retry_delay_sec = 5

    success = trigger_watch_webhook(config, watch, hits)

    assert success is False
    # No exception raised
