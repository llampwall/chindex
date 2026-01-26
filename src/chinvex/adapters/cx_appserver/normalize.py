# src/chinvex/adapters/cx_appserver/normalize.py
from __future__ import annotations

from .schemas import AppServerThread


def normalize_to_conversation_doc(thread: AppServerThread) -> dict:
    """
    Normalize AppServerThread to ConversationDoc internal schema.

    Returns dict matching ConversationDoc schema from P0 spec §2.
    """
    # Generate title if missing
    title = thread.title
    if not title and thread.turns:
        # Use first user message truncated
        for turn in thread.turns:
            if turn.role == "user" and turn.text:
                if len(turn.text) > 60:
                    title = turn.text[:57].strip() + "..."
                else:
                    title = turn.text.strip()
                break

    if not title:
        title = f"Thread {thread.id}"

    # Normalize turns
    normalized_turns = []
    for turn in thread.turns:
        normalized_turn = {
            "turn_id": turn.turn_id,
            "ts": turn.ts,
            "role": turn.role,
            "text": turn.text or "",
        }

        if turn.tool:
            normalized_turn["tool"] = turn.tool

        if turn.attachments:
            normalized_turn["attachments"] = turn.attachments

        if turn.meta:
            normalized_turn["meta"] = turn.meta

        normalized_turns.append(normalized_turn)

    return {
        "doc_type": "conversation",
        "source": "cx_appserver",
        "thread_id": thread.id,
        "title": title,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "turns": normalized_turns,
        "links": thread.links,
    }
