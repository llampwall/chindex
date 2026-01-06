from __future__ import annotations

from pathlib import Path

import typer

from .config import ConfigError, load_config
from .ingest import ingest
from .search import search
from .util import in_venv

app = typer.Typer(add_completion=False, help="chinvex: hybrid retrieval index CLI")


def _load_config(config_path: Path):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("ingest")
def ingest_cmd(
    config: Path = typer.Option(..., "--config", exists=True, help="Path to config JSON"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)
    cfg = _load_config(config)
    stats = ingest(cfg, ollama_host_override=ollama_host)
    typer.echo(f"Done. Documents: {stats['documents']} Chunks: {stats['chunks']} Skipped: {stats['skipped']}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    config: Path = typer.Option(..., "--config", exists=True, help="Path to config JSON"),
    k: int = typer.Option(8, "--k", help="Top K results"),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum score threshold"),
    source: str = typer.Option("all", "--source", help="all|repo|chat"),
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repo"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)
    if source not in {"all", "repo", "chat"}:
        raise typer.BadParameter("source must be one of: all, repo, chat")
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


if __name__ == "__main__":
    app()
