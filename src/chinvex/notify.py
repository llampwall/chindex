"""Push notification utilities (ntfy.sh)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


@dataclass
class NtfyConfig:
    """ntfy.sh notification configuration."""
    server: str  # e.g., "https://ntfy.sh"
    topic: str   # e.g., "chinvex-alerts"
    enabled: bool = True


def send_ntfy_push(
    config: NtfyConfig,
    message: str,
    title: str | None = None,
    priority: str = "default",
    tags: list[str] | None = None
) -> bool:
    """
    Send push notification via ntfy.sh.

    Args:
        config: ntfy configuration
        message: Notification body
        title: Optional title
        priority: Priority level (min, low, default, high, urgent)
        tags: Optional list of emoji tags

    Returns:
        True if sent successfully
    """
    if not config.enabled:
        log.debug("Notifications disabled, skipping")
        return False

    url = f"{config.server}/{config.topic}"

    headers = {}
    if title:
        headers["Title"] = title
    if priority != "default":
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            log.info(f"Sent ntfy push: {title or message[:30]}")
            return True
        else:
            log.warning(f"ntfy push failed: {response.status_code}")
            return False

    except requests.RequestException as e:
        log.error(f"Failed to send ntfy push: {e}")
        return False


def should_send_stale_alert(context_name: str, log_file: Path) -> bool:
    """
    Check if stale alert should be sent (dedup: max 1 per context per day).

    Args:
        context_name: Context to check
        log_file: Path to push_log.jsonl

    Returns:
        True if alert should be sent
    """
    if not log_file.exists():
        return True

    today = datetime.now(timezone.utc).date().isoformat()

    # Check log for existing entry
    try:
        for line in log_file.read_text().splitlines():
            record = json.loads(line)
            if (
                record.get("context") == context_name
                and record.get("type") == "stale"
                and record.get("date") == today
            ):
                return False  # Already sent today
    except (json.JSONDecodeError, IOError):
        pass

    return True


def log_push(context_name: str, push_type: str, log_file: Path) -> None:
    """
    Log push notification to dedup log.

    Args:
        context_name: Context name
        push_type: Type of push (stale, watch_hit, etc.)
        log_file: Path to push_log.jsonl
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    record = {
        "timestamp": now.isoformat(),
        "context": context_name,
        "type": push_type,
        "date": now.date().isoformat()
    }

    with log_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def send_stale_alert(context_name: str, log_file_path: str, server: str, topic: str) -> None:
    """
    Send stale context alert if not already sent today.

    Called from PowerShell sweep script.

    Args:
        context_name: Context name
        log_file_path: Path to push_log.jsonl
        server: ntfy server URL
        topic: ntfy topic
    """
    log_file = Path(log_file_path)

    # Check dedup
    if not should_send_stale_alert(context_name, log_file):
        log.debug(f"Stale alert for {context_name} already sent today")
        return

    # Send alert
    config = NtfyConfig(server=server, topic=topic, enabled=bool(topic))
    success = send_ntfy_push(
        config,
        f"{context_name}: last sync stale",
        title="Stale context",
        priority="low"
    )

    if success:
        # Log to dedup file
        log_push(context_name, "stale", log_file)
