from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    type: str
    path: Path
    name: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class AppConfig:
    index_dir: Path
    ollama_host: str
    embedding_model: str
    sources: tuple[SourceConfig, ...]


class ConfigError(ValueError):
    pass


def _expect_str(data: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config '{key}' must be a non-empty string.")
    return value.strip()


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a JSON object.")

    index_dir = _expect_str(raw, "index_dir")
    ollama_host = _expect_str(raw, "ollama_host")
    embedding_model = _expect_str(raw, "embedding_model")
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ConfigError("Config must include a non-empty 'sources' array.")

    sources: list[SourceConfig] = []
    for i, entry in enumerate(sources_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"Source entry {i} must be an object.")
        src_type = _expect_str(entry, "type")
        if src_type not in {"repo", "chat"}:
            raise ConfigError(f"Source entry {i} has invalid type '{src_type}'.")
        path_str = _expect_str(entry, "path")
        name = _expect_str(entry, "name", required=False)
        project = _expect_str(entry, "project", required=False)
        if src_type == "repo" and not name:
            raise ConfigError(f"Source entry {i} of type 'repo' requires 'name'.")
        if src_type == "chat" and not project:
            raise ConfigError(f"Source entry {i} of type 'chat' requires 'project'.")
        sources.append(
            SourceConfig(
                type=src_type,
                path=Path(path_str),
                name=name,
                project=project,
            )
        )

    return AppConfig(
        index_dir=Path(index_dir),
        ollama_host=ollama_host,
        embedding_model=embedding_model,
        sources=tuple(sources),
    )
