from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer

from .context import ContextConfig, list_contexts, load_context
from .storage import Storage
from .vectors import VectorStore


def get_contexts_root() -> Path:
    env_val = os.getenv("CHINVEX_CONTEXTS_ROOT")
    if env_val:
        return Path(env_val)
    return Path("P:/ai_memory/contexts")


def get_indexes_root() -> Path:
    env_val = os.getenv("CHINVEX_INDEXES_ROOT")
    if env_val:
        return Path(env_val)
    return Path("P:/ai_memory/indexes")


def create_context(name: str) -> None:
    """Create a new context with empty configuration."""
    # Validate name
    if not name or "/" in name or "\\" in name:
        typer.secho(f"Invalid context name: {name}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    contexts_root = get_contexts_root()
    indexes_root = get_indexes_root()

    ctx_dir = contexts_root / name
    if ctx_dir.exists():
        typer.secho(f"Context '{name}' already exists at {ctx_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Create directory structure
    ctx_dir.mkdir(parents=True, exist_ok=False)
    index_dir = indexes_root / name
    index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize context.json
    now = datetime.now(timezone.utc).isoformat()
    context_data = {
        "schema_version": 1,
        "name": name,
        "aliases": [],
        "includes": {
            "repos": [],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": str(index_dir / "hybrid.db"),
            "chroma_dir": str(index_dir / "chroma")
        },
        "weights": {
            "repo": 1.0,
            "chat": 0.8,
            "codex_session": 0.9,
            "note": 0.7
        },
        "created_at": now,
        "updated_at": now
    }

    context_file = ctx_dir / "context.json"
    context_file.write_text(json.dumps(context_data, indent=2), encoding="utf-8")

    # Initialize database
    db_path = index_dir / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    # Initialize Chroma
    chroma_dir = index_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    vectors = VectorStore(chroma_dir)
    # Just instantiate to create collection

    typer.secho(f"Created context: {name}", fg=typer.colors.GREEN)
    typer.echo(f"  Config: {context_file}")
    typer.echo(f"  Index:  {index_dir}")


def list_contexts_cli() -> None:
    """List all contexts."""
    contexts_root = get_contexts_root()

    from .context import list_contexts
    contexts = list_contexts(contexts_root)

    if not contexts:
        typer.echo("No contexts found.")
        return

    # Print table header
    typer.echo(f"{'NAME':<20} {'ALIASES':<30} {'UPDATED':<25}")
    typer.echo("-" * 75)

    for ctx in contexts:
        aliases_str = ", ".join(ctx.aliases) if ctx.aliases else "-"
        if len(aliases_str) > 28:
            aliases_str = aliases_str[:25] + "..."
        typer.echo(f"{ctx.name:<20} {aliases_str:<30} {ctx.updated_at:<25}")
