# src/chinvex/adapters/cx_appserver/client.py
from __future__ import annotations

import requests


class AppServerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def list_threads(self) -> list[dict]:
        """List all threads from /thread/list endpoint."""
        url = f"{self.base_url}/thread/list"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("threads", [])

    def get_thread(self, thread_id: str) -> dict:
        """Get full thread content from /thread/resume endpoint."""
        url = f"{self.base_url}/thread/resume/{thread_id}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
