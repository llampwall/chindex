from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer

from .context import ContextConfig, list_contexts, load_context
from .storage import Storage
from .util import backup_context_json, normalize_path_for_dedup
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


def create_context_if_missing(
    name: str,
    contexts_root: Path | None = None,
    repos: list[dict] | None = None,
    chat_roots: list[str] | None = None
) -> None:
    """
    Create context if it doesn't exist, with optional initial sources.
    Deduplicates paths before writing.

    repos format: [
        {
            "path": "P:/software/chinvex",
            "chinvex_depth": "full",
            "status": "active",
            "tags": ["python", "search"]
        }
    ]
    """
    if contexts_root is None:
        contexts_root = get_contexts_root()
    indexes_root = get_indexes_root()

    ctx_dir = contexts_root / name
    context_file = ctx_dir / "context.json"

    if context_file.exists():
        # Context exists, update with new sources if provided
        if repos or chat_roots:
            ctx_config = json.loads(context_file.read_text(encoding="utf-8"))

            # Normalize and deduplicate repos with metadata
            if repos:
                existing_repos = ctx_config["includes"].get("repos", [])
                # Build set of existing paths for deduplication
                seen = set()
                for r in existing_repos:
                    if isinstance(r, str):
                        seen.add(normalize_path_for_dedup(r))
                    else:
                        seen.add(normalize_path_for_dedup(r["path"]))

                for repo in repos:
                    normalized = normalize_path_for_dedup(repo["path"])
                    if normalized not in seen:
                        existing_repos.append({
                            "path": str(Path(repo["path"]).resolve()),
                            "chinvex_depth": repo["chinvex_depth"],
                            "status": repo["status"],
                            "tags": repo.get("tags", [])
                        })
                        seen.add(normalized)
                ctx_config["includes"]["repos"] = existing_repos

            # Normalize and deduplicate chat_roots
            if chat_roots:
                existing_chat_roots = ctx_config["includes"].get("chat_roots", [])
                seen = {normalize_path_for_dedup(r) for r in existing_chat_roots}
                for root in chat_roots:
                    normalized = normalize_path_for_dedup(root)
                    if normalized not in seen:
                        existing_chat_roots.append(str(Path(root).resolve()))
                        seen.add(normalized)
                ctx_config["includes"]["chat_roots"] = existing_chat_roots

            # Update timestamp
            ctx_config["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Write back
            backup_context_json(context_file)
            context_file.write_text(json.dumps(ctx_config, indent=2), encoding="utf-8")
        return  # Already exists

    # Create new context
    # Normalize and deduplicate paths with metadata
    normalized_repos = []
    if repos:
        seen = set()
        for repo in repos:
            normalized = normalize_path_for_dedup(repo["path"])
            if normalized not in seen:
                normalized_repos.append({
                    "path": str(Path(repo["path"]).resolve()),
                    "chinvex_depth": repo["chinvex_depth"],
                    "status": repo["status"],
                    "tags": repo.get("tags", [])
                })
                seen.add(normalized)

    normalized_chat_roots = []
    if chat_roots:
        seen = set()
        for root in chat_roots:
            normalized = normalize_path_for_dedup(root)
            if normalized not in seen:
                normalized_chat_roots.append(str(Path(root).resolve()))
                seen.add(normalized)

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
            "repos": normalized_repos,
            "chat_roots": normalized_chat_roots,
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
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        },
        "created_at": now,
        "updated_at": now
    }

    backup_context_json(context_file)
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
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        },
        "created_at": now,
        "updated_at": now
    }

    context_file = ctx_dir / "context.json"
    backup_context_json(context_file)
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


def sync_metadata_from_strap(
    context_name: str,
    registry_path: Path | None = None,
) -> dict:
    """
    Sync repo metadata from strap registry.json to chinvex context.json.

    Reads strap's registry.json and updates matching repos in context.json
    with current status, tags, and chinvex_depth values.

    Args:
        context_name: Name of the chinvex context to update
        registry_path: Path to registry.json (defaults to P:/software/_strap/registry.json)

    Returns:
        Dict with sync results: {"updated": [...], "not_found": [...]}
    """
    # Default registry path
    if registry_path is None:
        registry_path = Path("P:/software/_strap/registry.json")

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    # Load registry.json
    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_repos = registry_data.get("repos", [])

    # Build lookup map: normalized_path -> repo_entry
    registry_map = {}
    for repo_entry in registry_repos:
        repo_path = repo_entry.get("repoPath") or repo_entry.get("name")
        if repo_path:
            normalized = normalize_path_for_dedup(repo_path)
            registry_map[normalized] = repo_entry

    # Load context.json
    contexts_root = get_contexts_root()
    context_file = contexts_root / context_name / "context.json"

    if not context_file.exists():
        raise FileNotFoundError(f"Context not found: {context_name}")

    context_data = json.loads(context_file.read_text(encoding="utf-8"))

    # Update repo metadata
    updated = []
    not_found = []

    for repo in context_data["includes"].get("repos", []):
        # Handle both string and dict formats
        if isinstance(repo, str):
            repo_path = repo
            # Convert to dict format with defaults
            repo = {
                "path": repo_path,
                "chinvex_depth": "full",
                "status": "active",
                "tags": []
            }
        else:
            repo_path = repo["path"]

        normalized = normalize_path_for_dedup(repo_path)

        if normalized in registry_map:
            registry_entry = registry_map[normalized]
            old_values = {
                "depth": repo.get("chinvex_depth", "full"),
                "status": repo.get("status", "active"),
                "tags": repo.get("tags", [])
            }

            # Update from registry
            repo["chinvex_depth"] = registry_entry.get("chinvex_depth", "full")
            repo["status"] = registry_entry.get("status", "active")
            repo["tags"] = registry_entry.get("tags", [])

            new_values = {
                "depth": repo["chinvex_depth"],
                "status": repo["status"],
                "tags": repo["tags"]
            }

            if old_values != new_values:
                updated.append({
                    "path": repo_path,
                    "old": old_values,
                    "new": new_values
                })
        else:
            not_found.append(repo_path)

    # Update timestamp
    context_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Write back
    backup_context_json(context_file)
    context_file.write_text(json.dumps(context_data, indent=2), encoding="utf-8")

    return {
        "updated": updated,
        "not_found": not_found
    }
