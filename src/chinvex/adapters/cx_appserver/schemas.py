# src/chinvex/adapters/cx_appserver/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class AppServerTurn(BaseModel):
    """
    Schema for a turn from app-server API.

    Discovered from actual /thread/resume responses.
    """
    turn_id: str
    ts: str  # ISO8601
    role: str  # user|assistant|tool|system
    text: str | None = None
    tool: dict | None = None
    attachments: list[dict] = Field(default_factory=list)
    meta: dict | None = None


class AppServerThread(BaseModel):
    """
    Schema for a thread from app-server API.

    Discovered from actual /thread/list and /thread/resume responses.
    """
    id: str
    title: str | None = None
    created_at: str  # ISO8601
    updated_at: str  # ISO8601
    turns: list[AppServerTurn] = Field(default_factory=list)
    links: dict = Field(default_factory=dict)
