from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class ContextNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class ContextIncludes:
    repos: list[Path]
    chat_roots: list[Path]
    codex_session_roots: list[Path]
    note_roots: list[Path]


@dataclass(frozen=True)
class ContextIndex:
    sqlite_path: Path
    chroma_dir: Path


@dataclass(frozen=True)
class ContextConfig:
    schema_version: int
    name: str
    aliases: list[str]
    includes: ContextIncludes
    index: ContextIndex
    weights: dict[str, float]
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict) -> ContextConfig:
        includes_data = data["includes"]
        includes = ContextIncludes(
            repos=[Path(p) for p in includes_data.get("repos", [])],
            chat_roots=[Path(p) for p in includes_data.get("chat_roots", [])],
            codex_session_roots=[Path(p) for p in includes_data.get("codex_session_roots", [])],
            note_roots=[Path(p) for p in includes_data.get("note_roots", [])],
        )

        index_data = data["index"]
        index = ContextIndex(
            sqlite_path=Path(index_data["sqlite_path"]),
            chroma_dir=Path(index_data["chroma_dir"]),
        )

        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            aliases=data.get("aliases", []),
            includes=includes,
            index=index,
            weights=data["weights"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "aliases": self.aliases,
            "includes": {
                "repos": [str(p) for p in self.includes.repos],
                "chat_roots": [str(p) for p in self.includes.chat_roots],
                "codex_session_roots": [str(p) for p in self.includes.codex_session_roots],
                "note_roots": [str(p) for p in self.includes.note_roots],
            },
            "index": {
                "sqlite_path": str(self.index.sqlite_path),
                "chroma_dir": str(self.index.chroma_dir),
            },
            "weights": self.weights,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def load_context(name: str, contexts_root: Path) -> ContextConfig:
    """Load context by name or alias."""
    # Try direct name match first
    context_path = contexts_root / name / "context.json"
    if context_path.exists():
        data = json.loads(context_path.read_text(encoding="utf-8"))
        return ContextConfig.from_dict(data)

    # Try alias match
    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        context_file = ctx_dir / "context.json"
        if not context_file.exists():
            continue
        data = json.loads(context_file.read_text(encoding="utf-8"))
        aliases = data.get("aliases", [])
        if name in aliases:
            return ContextConfig.from_dict(data)

    raise ContextNotFoundError(
        f"Unknown context: {name}. Use 'chinvex context list' to see available contexts."
    )


def list_contexts(contexts_root: Path) -> list[ContextConfig]:
    """List all contexts, sorted by updated_at desc."""
    contexts: list[ContextConfig] = []

    if not contexts_root.exists():
        return contexts

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        context_file = ctx_dir / "context.json"
        if not context_file.exists():
            continue
        try:
            data = json.loads(context_file.read_text(encoding="utf-8"))
            contexts.append(ContextConfig.from_dict(data))
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by updated_at desc
    contexts.sort(key=lambda c: c.updated_at, reverse=True)
    return contexts
