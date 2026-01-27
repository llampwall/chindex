from __future__ import annotations

from pathlib import Path

import typer

from .config import ConfigError, load_config
from .context_cli import create_context, get_contexts_root, list_contexts_cli
from .ingest import ingest
from .search import search
from .util import in_venv

app = typer.Typer(add_completion=False, help="chinvex: hybrid retrieval index CLI")

# Add context subcommand group
context_app = typer.Typer(help="Manage contexts")
app.add_typer(context_app, name="context")

# Add state subcommand group
state_app = typer.Typer(help="Manage context state and STATE.md")
app.add_typer(state_app, name="state")


def _load_config(config_path: Path):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("ingest")
def ingest_cmd(
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to ingest"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if context:
        # New context-based ingestion
        from .context import load_context
        from .ingest import ingest_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)
        result = ingest_context(ctx, ollama_host_override=ollama_host)

        typer.secho(f"Ingestion complete for context '{context}':", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {result.stats['documents']}")
        typer.echo(f"  Chunks: {result.stats['chunks']}")
        typer.echo(f"  Skipped: {result.stats['skipped']}")
    else:
        # Old config-based ingestion (deprecated)
        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)

        cfg = _load_config(config)
        stats = ingest(cfg, ollama_host_override=ollama_host)
        typer.secho("Ingestion complete:", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {stats['documents']}")
        typer.echo(f"  Chunks: {stats['chunks']}")
        typer.echo(f"  Skipped: {stats['skipped']}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to search"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    k: int = typer.Option(8, "--k", help="Top K results"),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum score threshold"),
    source: str = typer.Option("all", "--source", help="all|repo|chat|codex_session"),
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repo"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if source not in {"all", "repo", "chat", "codex_session"}:
        raise typer.BadParameter("source must be one of: all, repo, chat, codex_session")

    if context:
        # New context-based search
        from .context import load_context
        from .search import search_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)

        results = search_context(
            ctx,
            query,
            k=k,
            min_score=min_score,
            source=source,
            project=project,
            repo=repo,
            ollama_host_override=ollama_host,
        )

        if not results:
            typer.echo("No results found.")
            return

        for i, result in enumerate(results, 1):
            typer.secho(f"\n[{i}] {result.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"Score: {result.score:.3f} | Type: {result.source_type}")
            typer.echo(f"Citation: {result.citation}")
            typer.echo(f"Snippet: {result.snippet}")
    else:
        # Old config-based search (deprecated)
        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)

        cfg = _load_config(config)
        results = search(
            cfg,
            query,
            k=k,
            min_score=min_score,
            source=source,
            project=project,
            repo=repo,
            ollama_host_override=ollama_host,
        )
        if not results:
            typer.echo("No results.")
            raise typer.Exit(code=0)

        for idx, result in enumerate(results, start=1):
            typer.echo(f"{idx}. score={result.score:.3f} source={result.source_type}")
            typer.echo(f"   {result.title}")
            typer.echo(f"   {result.citation}")
            typer.echo(f"   {result.snippet}")


@context_app.command("create")
def context_create_cmd(name: str = typer.Argument(..., help="Context name")) -> None:
    """Create a new context."""
    create_context(name)


@context_app.command("list")
def context_list_cmd() -> None:
    """List all contexts."""
    list_contexts_cli()


@state_app.command("generate")
def state_generate_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    llm: bool = typer.Option(False, "--llm", help="Enable LLM consolidation (P1.5)"),
    since: str = typer.Option("24h", "--since", help="Time window (e.g., 24h, 7d)"),
) -> None:
    """Generate state.json and STATE.md."""
    from datetime import datetime, timedelta, timezone
    from .context import load_context
    from .hooks import post_ingest_hook
    from .ingest import IngestRunResult

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Parse since duration (simple implementation)
    # TODO: implement full duration parsing
    if since.endswith("h"):
        hours = int(since[:-1])
        since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    elif since.endswith("d"):
        days = int(since[:-1])
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        typer.secho(f"Invalid --since format: {since}. Use format like '24h' or '7d'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Create fake result for manual generation
    result = IngestRunResult(
        run_id=f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        context=context,
        started_at=since_dt,
        finished_at=datetime.now(timezone.utc),
        new_doc_ids=[],
        updated_doc_ids=[],
        new_chunk_ids=[],
        skipped_doc_ids=[],
        error_doc_ids=[],
        stats={}
    )

    try:
        post_ingest_hook(ctx, result)
        typer.secho(f"Generated STATE.md for context '{context}'", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error generating state: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@state_app.command("show")
def state_show_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
) -> None:
    """Print STATE.md to stdout."""
    contexts_root = get_contexts_root()
    md_path = contexts_root / context / "STATE.md"

    if not md_path.exists():
        typer.secho(f"No STATE.md found for context '{context}'", fg=typer.colors.RED)
        typer.echo(f"Run 'chinvex state generate --context {context}' to create it.")
        raise typer.Exit(code=1)

    typer.echo(md_path.read_text(encoding='utf-8'))


if __name__ == "__main__":
    app()
