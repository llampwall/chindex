from __future__ import annotations

from dataclasses import dataclass


MAX_CHARS = 3000
REPO_OVERLAP = 300


@dataclass(frozen=True)
class Chunk:
    text: str
    ordinal: int
    char_start: int | None = None
    char_end: int | None = None
    msg_start: int | None = None
    msg_end: int | None = None
    roles_present: list[str] | None = None


def chunk_repo(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = MAX_CHARS - REPO_OVERLAP
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, ordinal=ordinal, char_start=start, char_end=end))
        ordinal += 1
        start += step
    if not chunks:
        chunks.append(Chunk(text="", ordinal=0, char_start=0, char_end=0))
    return chunks


def build_chat_text(messages: list[dict]) -> list[str]:
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "unknown"))
        text = str(msg.get("text", "")).strip()
        lines.append(f"[{i:04d}] {role}: {text}")
    return lines


def chunk_chat(messages: list[dict]) -> list[Chunk]:
    lines = build_chat_text(messages)
    chunks: list[Chunk] = []
    current: list[str] = []
    msg_start = 0
    ordinal = 0
    roles_present: set[str] = set()

    def flush(msg_end: int) -> None:
        nonlocal ordinal, msg_start, current, roles_present
        chunk_text = "\n".join(current)
        chunks.append(
            Chunk(
                text=chunk_text,
                ordinal=ordinal,
                msg_start=msg_start,
                msg_end=msg_end,
                roles_present=sorted(roles_present),
            )
        )
        ordinal += 1
        current = []
        roles_present = set()

    for i, line in enumerate(lines):
        msg = messages[i]
        role = str(msg.get("role", "unknown"))
        pending = current + [line]
        if len(pending) >= 10 or sum(len(x) + 1 for x in pending) > MAX_CHARS:
            if current:
                flush(i - 1)
                msg_start = i
            current.append(line)
            roles_present.add(role)
        else:
            current.append(line)
            roles_present.add(role)
            if len(current) == 10:
                flush(i)
                msg_start = i + 1

    if current:
        flush(len(lines) - 1)

    if not chunks:
        chunks.append(Chunk(text="", ordinal=0, msg_start=0, msg_end=0, roles_present=[]))
    return chunks
