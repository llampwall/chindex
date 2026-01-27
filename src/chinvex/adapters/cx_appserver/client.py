# src/chinvex/adapters/cx_appserver/client.py
from __future__ import annotations

import requests


class AppServerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> tuple[bool, str]:
        """
        Check if app-server is reachable.

        Returns:
            (success, message) tuple with detailed error info
        """
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 401:
                return (False, "Authentication failed (401). Check godex credentials.")
            if resp.status_code == 200:
                return (True, "App-server reachable")
            return (False, f"Unexpected status: {resp.status_code}")
        except requests.ConnectionError:
            return (False, f"Connection refused. Is app-server running at {self.base_url}?")
        except requests.Timeout:
            return (False, "Timeout connecting to app-server")
        except Exception as e:
            return (False, f"Unexpected error: {e}")

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
