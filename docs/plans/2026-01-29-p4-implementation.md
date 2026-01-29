# P4 Implementation Plan: Session Bootstrap + Daily Digest

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn Chinvex into a daily-usable memory service with proactive surfacing (digest + brief artifacts), pluggable embeddings (Ollama ⇄ OpenAI), and session bootstrap (Claude starts with context loaded).

**Architecture:**
- Embedding abstraction layer with provider protocol (Ollama, OpenAI)
- Index metadata system for dimension safety and provider tracking
- Deterministic digest/brief generators that consume watch history and memory files
- Memory file format (STATE.md, CONSTRAINTS.md, DECISIONS.md) in `docs/memory/`

**Tech Stack:** Python 3.12, typer CLI, OpenAI SDK, existing Ollama/Chroma/SQLite stack

---

## Phase 0: Setup Memory Files

### Task 0.1: Create memory file templates

**Files:**
- Create: `docs/memory/STATE.md`
- Create: `docs/memory/CONSTRAINTS.md`
- Create: `docs/memory/DECISIONS.md`
- Create: `tests/fixtures/memory/STATE.md`
- Create: `tests/fixtures/memory/CONSTRAINTS.md`
- Create: `tests/fixtures/memory/DECISIONS.md`

**Step 1: Create STATE.md template**

```bash
mkdir -p docs/memory
```

Create `docs/memory/STATE.md`:
```markdown
# State

## Current Objective
P4 implementation - session bootstrap + daily digest

## Active Work
- Setting up memory file structure
- Preparing for embedding abstraction

## Blockers
None

## Next Actions
- [ ] Implement inline context creation (P4.1)
- [ ] Add embedding provider abstraction (P4.2)
```

**Step 2: Create CONSTRAINTS.md template**

Create `docs/memory/CONSTRAINTS.md`:
```markdown
# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors
- Embedding dims locked per index (see meta.json)
- Gateway port: 7778
- Contexts root: P:\ai_memory\contexts

## Rules
- Schema stays v2 - no migrations without rebuild
- Metrics endpoint requires auth
- Archive is dry-run by default
- Index metadata (meta.json) is source of truth for dimensions

## Key Facts
- Gateway: localhost:7778 → chinvex.unkndlabs.com
- Token env var: CHINVEX_API_TOKEN
- OpenAI API key: OPENAI_API_KEY
```

**Step 3: Create DECISIONS.md template**

Create `docs/memory/DECISIONS.md`:
```markdown
# Decisions

### 2026-01-29 — Created memory file structure

- **Why:** P4 requires persistent state tracking for digest/brief generation
- **Impact:** Claude can now load context from STATE.md on session start
- **Evidence:** `docs/memory/` directory
```

**Step 4: Create test fixtures**

```bash
mkdir -p tests/fixtures/memory
cp docs/memory/STATE.md tests/fixtures/memory/STATE.md
cp docs/memory/CONSTRAINTS.md tests/fixtures/memory/CONSTRAINTS.md
cp docs/memory/DECISIONS.md tests/fixtures/memory/DECISIONS.md
```

**Step 5: Commit**

```bash
git add docs/memory/ tests/fixtures/memory/
git commit -m "docs: add memory file structure (P4.0)

Created STATE.md, CONSTRAINTS.md, DECISIONS.md templates
for session bootstrap and digest generation.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 1: Inline Context Creation (P4.1)

### Task 1.1: Path normalization utility

**Files:**
- Modify: `src/chinvex/util.py` (add functions)
- Create: `tests/test_path_normalization.py`

**Step 1: Write the failing test**

Create `tests/test_path_normalization.py`:
```python
from pathlib import Path
from chinvex.util import normalize_path_for_dedup

def test_normalize_path_converts_to_absolute():
    result = normalize_path_for_dedup("./src")
    assert result.is_absolute()

def test_normalize_path_uses_forward_slashes():
    result = normalize_path_for_dedup(r"C:\Code\chinvex")
    assert "\\" not in result

def test_normalize_path_deduplicates_same_path():
    path1 = normalize_path_for_dedup("./src")
    path2 = normalize_path_for_dedup(Path.cwd() / "src")
    assert path1 == path2

def test_normalize_path_case_insensitive_on_windows():
    import platform
    if platform.system() != "Windows":
        return  # Skip on non-Windows
    path1 = normalize_path_for_dedup(r"C:\Code\Chinvex")
    path2 = normalize_path_for_dedup(r"c:\code\chinvex")
    assert path1 == path2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_path_normalization.py -v`
Expected: FAIL with "ImportError: cannot import name 'normalize_path_for_dedup'"

**Step 3: Write minimal implementation**

Add to `src/chinvex/util.py`:
```python
import platform
from pathlib import Path

def normalize_path_for_dedup(path: str | Path) -> str:
    """
    Normalize path for deduplication:
    - Convert to absolute
    - Use forward slashes
    - Lowercase on Windows (case-insensitive)
    """
    abs_path = Path(path).resolve()
    normalized = abs_path.as_posix()

    if platform.system() == "Windows":
        normalized = normalized.lower()

    return normalized
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_path_normalization.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/chinvex/util.py tests/test_path_normalization.py
git commit -m "feat(util): add path normalization for deduplication (P4.1)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 1.2: Inline context creation in ingest

**Files:**
- Modify: `src/chinvex/cli.py:38-76` (ingest_cmd function)
- Modify: `src/chinvex/context_cli.py` (add create_if_missing)
- Create: `tests/test_inline_context_creation.py`

**Step 1: Write the failing test**

Create `tests/test_inline_context_creation.py`:
```python
import pytest
from pathlib import Path
from chinvex.context import load_context
from chinvex.context_cli import get_contexts_root

def test_ingest_creates_context_if_missing(tmp_path, monkeypatch):
    """Test that ingest auto-creates context with --repo flag."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "test.txt").write_text("hello")

    # Context doesn't exist yet
    assert not (contexts_root / "NewContext" / "context.json").exists()

    # Run ingest with --repo
    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    result = runner.invoke(app, [
        "ingest",
        "--context", "NewContext",
        "--repo", str(repo_path)
    ])

    assert result.exit_code == 0
    assert (contexts_root / "NewContext" / "context.json").exists()

    # Verify context.json has repo in includes
    ctx = load_context("NewContext", contexts_root)
    assert len(ctx.includes.repos) == 1

def test_ingest_deduplicates_repo_paths(tmp_path, monkeypatch):
    """Test that duplicate --repo paths are ignored."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "test.txt").write_text("hello")

    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    # Run ingest with duplicate --repo (different forms)
    result = runner.invoke(app, [
        "ingest",
        "--context", "NewContext",
        "--repo", str(repo_path),
        "--repo", str(repo_path.resolve())  # Same path, different form
    ])

    assert result.exit_code == 0
    ctx = load_context("NewContext", contexts_root)
    assert len(ctx.includes.repos) == 1  # Deduplicated

def test_ingest_fails_if_repo_doesnt_exist(tmp_path, monkeypatch):
    """Test that --repo path that doesn't exist fails immediately."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    result = runner.invoke(app, [
        "ingest",
        "--context", "NewContext",
        "--repo", "/nonexistent/path"
    ])

    assert result.exit_code != 0
    assert "does not exist" in result.stdout.lower()

def test_ingest_no_write_context_flag(tmp_path, monkeypatch):
    """Test --no-write-context ingests without persisting context.json."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "test.txt").write_text("hello")

    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    # Run with --no-write-context
    result = runner.invoke(app, [
        "ingest",
        "--context", "TempContext",
        "--repo", str(repo_path),
        "--no-write-context"
    ])

    # Should succeed but not create context.json
    assert result.exit_code == 0
    assert not (contexts_root / "TempContext" / "context.json").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_inline_context_creation.py -v`
Expected: FAIL (tests fail because functionality not implemented)

**Step 3: Implement context auto-creation**

Modify `src/chinvex/context_cli.py`, add function:
```python
from .util import normalize_path_for_dedup

def create_context_if_missing(
    name: str,
    contexts_root: Path,
    repos: list[str] | None = None,
    chat_roots: list[str] | None = None
) -> None:
    """
    Create context if it doesn't exist, with optional initial sources.
    Deduplicates paths before writing.
    """
    context_dir = contexts_root / name
    context_file = context_dir / "context.json"

    if context_file.exists():
        return  # Already exists, don't clobber

    # Normalize and deduplicate paths
    normalized_repos = []
    if repos:
        seen = set()
        for repo in repos:
            normalized = normalize_path_for_dedup(repo)
            if normalized not in seen:
                normalized_repos.append(Path(repo).resolve())
                seen.add(normalized)

    normalized_chat_roots = []
    if chat_roots:
        seen = set()
        for root in chat_roots:
            normalized = normalize_path_for_dedup(root)
            if normalized not in seen:
                normalized_chat_roots.append(Path(root).resolve())
                seen.add(normalized)

    # Create context
    create_context(name, contexts_root)

    # Update with sources if provided
    if normalized_repos or normalized_chat_roots:
        ctx_config = json.loads(context_file.read_text())
        if normalized_repos:
            ctx_config["includes"]["repos"] = [str(p) for p in normalized_repos]
        if normalized_chat_roots:
            ctx_config["includes"]["chat_roots"] = [str(p) for p in normalized_chat_roots]
        context_file.write_text(json.dumps(ctx_config, indent=2))
```

**Step 4: Update CLI ingest command**

Modify `src/chinvex/cli.py:38-76`:
```python
@app.command("ingest")
def ingest_cmd(
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to ingest"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
    rechunk_only: bool = typer.Option(False, "--rechunk-only", help="Rechunk only, reuse embeddings when possible"),
    repo: list[str] = typer.Option([], "--repo", help="Add repo path to context (can be repeated)"),
    chat_root: list[str] = typer.Option([], "--chat-root", help="Add chat root to context (can be repeated)"),
    no_write_context: bool = typer.Option(False, "--no-write-context", help="Ingest ad-hoc without mutating context.json"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if context:
        # Validate repo paths exist
        for repo_path in repo:
            if not Path(repo_path).exists():
                typer.secho(f"Error: Repo path does not exist: {repo_path}", fg=typer.colors.RED)
                raise typer.Exit(code=2)

        # Validate chat roots exist
        for chat_path in chat_root:
            if not Path(chat_path).exists():
                typer.secho(f"Error: Chat root does not exist: {chat_path}", fg=typer.colors.RED)
                raise typer.Exit(code=2)

        from .context import load_context
        from .context_cli import create_context_if_missing, get_contexts_root
        from .ingest import ingest_context

        contexts_root = get_contexts_root()

        # Auto-create context if needed (unless --no-write-context)
        if not no_write_context:
            create_context_if_missing(
                context,
                contexts_root,
                repos=repo if repo else None,
                chat_roots=chat_root if chat_root else None
            )

        ctx = load_context(context, contexts_root)
        result = ingest_context(ctx, ollama_host_override=ollama_host, rechunk_only=rechunk_only)

        typer.secho(f"Ingestion complete for context '{context}':", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {result.stats['documents']}")
        typer.echo(f"  Chunks: {result.stats['chunks']}")
        typer.echo(f"  Skipped: {result.stats['skipped']}")
        if 'embeddings_reused' in result.stats:
            typer.echo(f"  Embeddings: {result.stats['embeddings_reused']} reused, {result.stats['embeddings_new']} new")
    else:
        # Old config-based ingestion (deprecated)
        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)
        cfg = _load_config(config)
        stats = ingest(cfg, ollama_host_override=ollama_host)
        typer.secho("Ingestion complete:", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {stats['documents']}")
        typer.echo(f"  Chunks: {stats['chunks']}")
        typer.echo(f"  Skipped: {stats['skipped']}")
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_inline_context_creation.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/context_cli.py tests/test_inline_context_creation.py
git commit -m "feat(cli): add inline context creation with --repo flag (P4.1)

Auto-creates context if it doesn't exist during ingest.
Deduplicates paths and validates existence before ingesting.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Embedding Provider Abstraction (P4.2)

### Task 2.1: Embedding provider protocol

**Files:**
- Create: `src/chinvex/embedding_providers.py`
- Create: `tests/test_embedding_providers.py`

**Step 1: Write the failing test**

Create `tests/test_embedding_providers.py`:
```python
import pytest
from chinvex.embedding_providers import EmbeddingProvider, OllamaProvider, OpenAIProvider

def test_embedding_provider_protocol():
    """Test that providers implement the protocol."""
    def accepts_provider(provider: EmbeddingProvider):
        _ = provider.dimensions
        _ = provider.model_name
        _ = provider.embed(["test"])

    # This should type-check with mypy

def test_ollama_provider_dimensions():
    provider = OllamaProvider("http://localhost:11434", "mxbai-embed-large")
    assert provider.dimensions == 1024
    assert provider.model_name == "mxbai-embed-large"

def test_openai_provider_dimensions():
    provider = OpenAIProvider(api_key="test", model="text-embedding-3-small")
    assert provider.dimensions == 1536
    assert provider.model_name == "text-embedding-3-small"

def test_openai_provider_requires_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key=None, model="text-embedding-3-small")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding_providers.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `src/chinvex/embedding_providers.py`:
```python
from __future__ import annotations

import logging
import os
from typing import Protocol

log = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class OllamaProvider:
    """Ollama embedding provider."""

    # Model dimensions (hardcoded for now, could query Ollama API)
    MODEL_DIMS = {
        "mxbai-embed-large": 1024,
    }

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown Ollama model: {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Implementation will use existing OllamaEmbedder
        from .embed import OllamaEmbedder
        embedder = OllamaEmbedder(self.host, self.model)
        return embedder.embed(texts)

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS[self.model]

    @property
    def model_name(self) -> str:
        return self.model


class OpenAIProvider:
    """OpenAI embedding provider."""

    # Model dimensions
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown OpenAI model: {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Will implement OpenAI API call in next task
        raise NotImplementedError("OpenAI embedding not yet implemented")

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS[self.model]

    @property
    def model_name(self) -> str:
        return self.model
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedding_providers.py::test_embedding_provider_protocol -v`
Run: `pytest tests/test_embedding_providers.py::test_ollama_provider_dimensions -v`
Run: `pytest tests/test_embedding_providers.py::test_openai_provider_dimensions -v`
Run: `pytest tests/test_embedding_providers.py::test_openai_provider_requires_api_key -v`
Expected: PASS (except embed test which expects NotImplementedError)

**Step 5: Commit**

```bash
git add src/chinvex/embedding_providers.py tests/test_embedding_providers.py
git commit -m "feat(embed): add embedding provider protocol (P4.2)

Defines EmbeddingProvider protocol and initial OllamaProvider,
OpenAIProvider implementations.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: OpenAI embedding implementation

**Files:**
- Modify: `src/chinvex/embedding_providers.py:50-55` (OpenAIProvider.embed)
- Modify: `pyproject.toml` (add openai dependency)
- Create: `tests/test_openai_embeddings.py`

**Step 1: Add openai dependency**

Modify `pyproject.toml`:
```toml
dependencies = [
  "typer>=0.12.3",
  "chromadb>=0.5.3",
  "requests>=2.32.3",
  "mcp>=1.0.0",
  "portalocker>=2.10.1",
  "fastapi>=0.109.0",
  "uvicorn[standard]>=0.27.0",
  "pydantic>=2.5.0",
  "httpx>=0.27.0",
  "openai>=1.12.0",
]
```

Install: `pip install -e .`

**Step 2: Write the failing test (with mocking)**

Create `tests/test_openai_embeddings.py`:
```python
import pytest
from unittest.mock import Mock, patch
from chinvex.embedding_providers import OpenAIProvider

def test_openai_embed_calls_api():
    """Test OpenAI embedding makes API call with correct params."""
    provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")

    with patch("openai.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3] * 512)  # 1536 dims
        ]
        mock_client.embeddings.create.return_value = mock_response

        result = provider.embed(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_client.embeddings.create.assert_called_once()

def test_openai_embed_batching():
    """Test OpenAI respects batch size limits."""
    provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")

    texts = ["text"] * 3000  # Exceeds batch limit of 2048

    with patch("openai.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(2048)]
        mock_client.embeddings.create.return_value = mock_response

        result = provider.embed(texts)

        # Should make 2 API calls (2048 + 952)
        assert mock_client.embeddings.create.call_count == 2

def test_openai_embed_retry_on_rate_limit():
    """Test OpenAI retries on rate limit errors."""
    provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")

    with patch("openai.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # First call raises rate limit, second succeeds
        from openai import RateLimitError
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]

        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(status_code=429), body=None),
            mock_response
        ]

        result = provider.embed(["test"])

        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 2
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_openai_embeddings.py -v`
Expected: FAIL (NotImplementedError)

**Step 4: Implement OpenAI embedding**

Modify `src/chinvex/embedding_providers.py`, update OpenAIProvider.embed:
```python
import time
from openai import OpenAI, RateLimitError, APIError

class OpenAIProvider:
    # ... existing code ...

    MAX_BATCH_SIZE = 2048
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown OpenAI model: {model}")

        self.client = OpenAI(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using OpenAI API.
        Handles batching (max 2048 texts) and retries (3x with backoff).
        """
        all_embeddings = []

        # Batch texts
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    log.warning(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"OpenAI rate limit exceeded after {self.MAX_RETRIES} attempts") from e
            except APIError as e:
                raise RuntimeError(f"OpenAI API error: {e}") from e
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_openai_embeddings.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/chinvex/embedding_providers.py pyproject.toml tests/test_openai_embeddings.py
git commit -m "feat(embed): implement OpenAI embedding provider (P4.2)

Adds OpenAI API client with batching (2048 limit) and retry
logic (3x exponential backoff). Mocked tests validate behavior.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2.3: Index metadata system

**Files:**
- Create: `src/chinvex/index_meta.py`
- Create: `tests/test_index_meta.py`

**Step 1: Write the failing test**

Create `tests/test_index_meta.py`:
```python
import pytest
from pathlib import Path
from chinvex.index_meta import IndexMeta, read_index_meta, write_index_meta

def test_index_meta_creation(tmp_path):
    """Test creating index metadata."""
    meta = IndexMeta(
        schema_version=2,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        created_at="2026-01-29T12:00:00Z"
    )

    assert meta.embedding_provider == "openai"
    assert meta.embedding_dimensions == 1536

def test_write_and_read_index_meta(tmp_path):
    """Test writing and reading index metadata."""
    meta_path = tmp_path / "meta.json"

    meta = IndexMeta(
        schema_version=2,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-29T12:00:00Z"
    )

    write_index_meta(meta_path, meta)
    assert meta_path.exists()

    loaded = read_index_meta(meta_path)
    assert loaded.embedding_provider == "ollama"
    assert loaded.embedding_dimensions == 1024

def test_read_missing_meta_returns_none(tmp_path):
    """Test reading non-existent meta.json returns None."""
    meta_path = tmp_path / "meta.json"
    result = read_index_meta(meta_path)
    assert result is None

def test_dimension_mismatch_check():
    """Test checking dimension mismatch."""
    meta = IndexMeta(
        schema_version=2,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-29T12:00:00Z"
    )

    # Match
    assert meta.matches_provider("ollama", "mxbai-embed-large", 1024) is True

    # Mismatch on dimensions
    assert meta.matches_provider("ollama", "mxbai-embed-large", 1536) is False

    # Mismatch on provider
    assert meta.matches_provider("openai", "text-embedding-3-small", 1024) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_index_meta.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `src/chinvex/index_meta.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexMeta:
    """Index metadata tracking embedding provider and dimensions."""
    schema_version: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    created_at: str

    def matches_provider(
        self,
        provider: str,
        model: str,
        dimensions: int
    ) -> bool:
        """Check if provider/model/dims match this index."""
        return (
            self.embedding_provider == provider and
            self.embedding_model == model and
            self.embedding_dimensions == dimensions
        )


def read_index_meta(path: Path) -> IndexMeta | None:
    """Read index metadata from meta.json, or None if missing."""
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    return IndexMeta(
        schema_version=data["schema_version"],
        embedding_provider=data["embedding_provider"],
        embedding_model=data["embedding_model"],
        embedding_dimensions=data["embedding_dimensions"],
        created_at=data["created_at"]
    )


def write_index_meta(path: Path, meta: IndexMeta) -> None:
    """Write index metadata to meta.json."""
    data = {
        "schema_version": meta.schema_version,
        "embedding_provider": meta.embedding_provider,
        "embedding_model": meta.embedding_model,
        "embedding_dimensions": meta.embedding_dimensions,
        "created_at": meta.created_at
    }
    path.write_text(json.dumps(data, indent=2))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_index_meta.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/chinvex/index_meta.py tests/test_index_meta.py
git commit -m "feat(storage): add index metadata system (P4.2)

Tracks embedding provider, model, and dimensions in meta.json
to enforce consistency and prevent provider mixing.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2.4: Provider selection in ingest

**Files:**
- Modify: `src/chinvex/ingest.py:41-75` (ingest function)
- Modify: `src/chinvex/cli.py:38-76` (add --embed-provider flag)
- Create: `tests/test_provider_selection.py`

**Step 1: Write the failing test**

Create `tests/test_provider_selection.py`:
```python
import pytest
import os
from pathlib import Path
from chinvex.embedding_providers import get_provider
from chinvex.context import ContextConfig

def test_provider_selection_precedence_cli():
    """Test CLI flag takes precedence."""
    provider = get_provider(
        cli_provider="openai",
        context_config=None,
        env_provider="ollama"
    )
    assert provider.model_name == "text-embedding-3-small"

def test_provider_selection_precedence_context():
    """Test context.json takes precedence over env."""
    context_config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        }
    }
    provider = get_provider(
        cli_provider=None,
        context_config=context_config,
        env_provider="ollama"
    )
    assert provider.model_name == "text-embedding-3-small"

def test_provider_selection_precedence_env():
    """Test env var used if CLI and context not set."""
    provider = get_provider(
        cli_provider=None,
        context_config=None,
        env_provider="openai"
    )
    assert provider.model_name == "text-embedding-3-small"

def test_provider_selection_default_ollama():
    """Test default is ollama."""
    provider = get_provider(
        cli_provider=None,
        context_config=None,
        env_provider=None
    )
    assert provider.model_name == "mxbai-embed-large"

def test_dimension_mismatch_fails(tmp_path):
    """Test that dimension mismatch fails ingest."""
    # Create meta.json with ollama dims
    from chinvex.index_meta import IndexMeta, write_index_meta
    meta_path = tmp_path / "meta.json"
    meta = IndexMeta(
        schema_version=2,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-29T12:00:00Z"
    )
    write_index_meta(meta_path, meta)

    # Try to ingest with openai (different dims)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        # validate_provider_match will raise
        pass
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider_selection.py -v`
Expected: FAIL with "ImportError: cannot import name 'get_provider'"

**Step 3: Implement provider selection logic**

Add to `src/chinvex/embedding_providers.py`:
```python
def get_provider(
    cli_provider: str | None,
    context_config: dict | None,
    env_provider: str | None,
    ollama_host: str = "http://localhost:11434"
) -> EmbeddingProvider:
    """
    Select embedding provider based on precedence:
    1. CLI flag
    2. context.json
    3. Environment variable
    4. Default (ollama)
    """
    provider_name = None
    model = None

    # 1. CLI
    if cli_provider:
        provider_name = cli_provider
    # 2. context.json
    elif context_config and "embedding" in context_config:
        provider_name = context_config["embedding"].get("provider")
        model = context_config["embedding"].get("model")
    # 3. Environment
    elif env_provider:
        provider_name = env_provider
    # 4. Default
    else:
        provider_name = "ollama"

    # Instantiate provider
    if provider_name == "ollama":
        model = model or "mxbai-embed-large"
        return OllamaProvider(ollama_host, model)
    elif provider_name == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIProvider(api_key=None, model=model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
```

**Step 4: Update ingest to use provider selection**

Modify `src/chinvex/ingest.py`, update `ingest_context` function:
```python
def ingest_context(
    ctx: ContextConfig,
    *,
    ollama_host_override: str | None = None,
    rechunk_only: bool = False,
    embed_provider: str | None = None,
    rebuild_index: bool = False,
) -> IngestRunResult:
    """Ingest sources for a context."""
    from .embedding_providers import get_provider
    from .index_meta import IndexMeta, read_index_meta, write_index_meta
    from .util import now_iso

    # Get provider
    env_provider = os.getenv("CHINVEX_EMBED_PROVIDER")
    provider = get_provider(
        cli_provider=embed_provider,
        context_config=None,  # TODO: read from ctx config
        env_provider=env_provider,
        ollama_host=ollama_host_override or ctx.ollama.base_url
    )

    # Check index metadata
    index_dir = ctx.index.sqlite_path.parent
    meta_path = index_dir / "meta.json"
    existing_meta = read_index_meta(meta_path)

    if existing_meta:
        # Validate dimensions match
        if not existing_meta.matches_provider(
            provider.__class__.__name__.replace("Provider", "").lower(),
            provider.model_name,
            provider.dimensions
        ):
            if not rebuild_index:
                raise RuntimeError(
                    f"Dimension mismatch: index uses {existing_meta.embedding_provider} "
                    f"({existing_meta.embedding_dimensions}D) but provider is "
                    f"{provider.model_name} ({provider.dimensions}D). "
                    f"Use --rebuild-index to switch providers."
                )
    else:
        # Create meta.json
        meta = IndexMeta(
            schema_version=2,
            embedding_provider=provider.__class__.__name__.replace("Provider", "").lower(),
            embedding_model=provider.model_name,
            embedding_dimensions=provider.dimensions,
            created_at=now_iso()
        )
        write_index_meta(meta_path, meta)

    # Continue with existing ingest logic...
```

**Step 5: Add --embed-provider flag to CLI**

Modify `src/chinvex/cli.py`:
```python
@app.command("ingest")
def ingest_cmd(
    # ... existing params ...
    embed_provider: str | None = typer.Option(None, "--embed-provider", help="Embedding provider: ollama|openai"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Rebuild index (wipe and re-embed)"),
) -> None:
    # ... existing logic ...
    result = ingest_context(
        ctx,
        ollama_host_override=ollama_host,
        rechunk_only=rechunk_only,
        embed_provider=embed_provider,
        rebuild_index=rebuild_index
    )
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_provider_selection.py -v`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add src/chinvex/embedding_providers.py src/chinvex/ingest.py src/chinvex/cli.py tests/test_provider_selection.py
git commit -m "feat(ingest): add provider selection with precedence (P4.2)

Implements provider selection: CLI > context.json > env > default.
Validates dimensions against index meta.json to prevent mixing.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Digest Generation (P4.3)

### Task 3.1: Ingest run log

**Files:**
- Modify: `src/chinvex/ingest.py:26-75` (add run logging)
- Create: `src/chinvex/ingest_log.py`
- Create: `tests/test_ingest_log.py`

**Step 1: Write the failing test**

Create `tests/test_ingest_log.py`:
```python
import pytest
from pathlib import Path
from chinvex.ingest_log import log_run_start, log_run_end, read_ingest_runs

def test_log_run_start(tmp_path):
    """Test logging run start."""
    log_path = tmp_path / "ingest_runs.jsonl"
    run_id = "test-run-123"

    log_run_start(log_path, run_id, sources=["repo1", "repo2"])

    assert log_path.exists()
    runs = read_ingest_runs(log_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "started"

def test_log_run_end_success(tmp_path):
    """Test logging successful run end."""
    log_path = tmp_path / "ingest_runs.jsonl"
    run_id = "test-run-123"

    log_run_start(log_path, run_id, sources=["repo1"])
    log_run_end(
        log_path,
        run_id,
        status="succeeded",
        docs_seen=100,
        docs_changed=10,
        chunks_new=50,
        chunks_updated=20
    )

    runs = read_ingest_runs(log_path)
    assert len(runs) == 2
    assert runs[1]["status"] == "succeeded"
    assert runs[1]["docs_seen"] == 100

def test_log_run_end_failure(tmp_path):
    """Test logging failed run end."""
    log_path = tmp_path / "ingest_runs.jsonl"
    run_id = "test-run-123"

    log_run_start(log_path, run_id, sources=["repo1"])
    log_run_end(
        log_path,
        run_id,
        status="failed",
        error="OpenAI rate limit exceeded"
    )

    runs = read_ingest_runs(log_path)
    assert len(runs) == 2
    assert runs[1]["status"] == "failed"
    assert "rate limit" in runs[1]["error"]

def test_read_completed_runs_only(tmp_path):
    """Test filtering for completed runs only."""
    log_path = tmp_path / "ingest_runs.jsonl"

    # Run 1: completed
    log_run_start(log_path, "run1", sources=["repo1"])
    log_run_end(log_path, "run1", status="succeeded", docs_seen=10)

    # Run 2: started but not completed (crash)
    log_run_start(log_path, "run2", sources=["repo1"])

    # Run 3: completed
    log_run_start(log_path, "run3", sources=["repo1"])
    log_run_end(log_path, "run3", status="succeeded", docs_seen=20)

    completed = read_ingest_runs(log_path, completed_only=True)
    assert len(completed) == 2
    assert completed[0]["run_id"] == "run1"
    assert completed[1]["run_id"] == "run3"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_log.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `src/chinvex/ingest_log.py`:
```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def log_run_start(
    log_path: Path,
    run_id: str,
    sources: list[str]
) -> None:
    """Log the start of an ingest run."""
    entry = {
        "run_id": run_id,
        "status": "started",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "sources": sources
    }

    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def log_run_end(
    log_path: Path,
    run_id: str,
    status: str,  # "succeeded" or "failed"
    docs_seen: int = 0,
    docs_changed: int = 0,
    chunks_new: int = 0,
    chunks_updated: int = 0,
    error: str | None = None
) -> None:
    """Log the end of an ingest run."""
    entry = {
        "run_id": run_id,
        "status": status,
        "ended_at": datetime.utcnow().isoformat() + "Z"
    }

    if status == "succeeded":
        entry.update({
            "docs_seen": docs_seen,
            "docs_changed": docs_changed,
            "chunks_new": chunks_new,
            "chunks_updated": chunks_updated
        })
    elif status == "failed":
        entry["error"] = error

    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_ingest_runs(
    log_path: Path,
    completed_only: bool = False
) -> list[dict]:
    """
    Read ingest runs from JSONL log.
    If completed_only=True, filter out runs without matching end record.
    """
    if not log_path.exists():
        return []

    runs = []
    with log_path.open("r") as f:
        for line in f:
            runs.append(json.loads(line.strip()))

    if not completed_only:
        return runs

    # Filter: keep only runs with succeeded/failed status
    completed_run_ids = {
        r["run_id"] for r in runs
        if r["status"] in ("succeeded", "failed")
    }

    return [r for r in runs if r["run_id"] in completed_run_ids]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_log.py -v`
Expected: PASS (all tests)

**Step 5: Integrate into ingest**

Modify `src/chinvex/ingest.py`, add logging:
```python
from .ingest_log import log_run_start, log_run_end

def ingest_context(
    ctx: ContextConfig,
    # ... params ...
) -> IngestRunResult:
    # ... existing setup ...

    # Generate run ID
    import uuid
    run_id = str(uuid.uuid4())

    # Log run start
    log_path = ctx.index.sqlite_path.parent.parent / ctx.name / "ingest_runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_run_start(log_path, run_id, sources=[str(p) for p in ctx.includes.repos])

    try:
        # ... existing ingest logic ...

        # Log run end (success)
        log_run_end(
            log_path,
            run_id,
            status="succeeded",
            docs_seen=stats["documents"] + stats["skipped"],
            docs_changed=stats["documents"],
            chunks_new=stats["chunks"],
            chunks_updated=0  # TODO: track updates
        )

        return result
    except Exception as e:
        # Log run end (failure)
        log_run_end(log_path, run_id, status="failed", error=str(e))
        raise
```

**Step 6: Run integration test**

Create small test that runs ingest and checks log:
```python
def test_ingest_logs_runs(tmp_path, monkeypatch):
    # ... setup context ...
    # ... run ingest ...
    # ... verify ingest_runs.jsonl exists and has entries ...
```

**Step 7: Commit**

```bash
git add src/chinvex/ingest_log.py src/chinvex/ingest.py tests/test_ingest_log.py
git commit -m "feat(ingest): add run logging to ingest_runs.jsonl (P4.3)

Logs start/end of each ingest run with stats and error tracking.
Supports filtering for completed runs only (ignores crashes).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3.2: Digest CLI command

**Files:**
- Modify: `src/chinvex/cli.py` (add digest command)
- Create: `src/chinvex/digest.py`
- Create: `tests/test_digest.py`

**Step 1: Write the failing test**

Create `tests/test_digest.py`:
```python
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from chinvex.digest import generate_digest

def test_generate_digest_basic(tmp_path):
    """Test basic digest generation."""
    # Setup: create ingest_runs.jsonl
    runs_log = tmp_path / "ingest_runs.jsonl"
    runs_log.write_text(
        '{"run_id": "run1", "status": "started", "started_at": "2026-01-29T12:00:00Z"}\n'
        '{"run_id": "run1", "status": "succeeded", "ended_at": "2026-01-29T12:05:00Z", "docs_seen": 100, "docs_changed": 10, "chunks_new": 50, "chunks_updated": 20}\n'
    )

    # Generate digest
    output_md = tmp_path / "digest.md"
    output_json = tmp_path / "digest.json"

    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=None,
        state_md=None,
        output_md=output_md,
        output_json=output_json,
        since_hours=24
    )

    assert output_md.exists()
    assert output_json.exists()

    # Check markdown content
    content = output_md.read_text()
    assert "# Digest:" in content
    assert "10" in content  # docs_changed
    assert "50" in content  # chunks_new

def test_generate_digest_with_watches(tmp_path):
    """Test digest includes watch hits."""
    # Setup watch history
    watch_log = tmp_path / "watch_history.jsonl"
    watch_log.write_text(
        '{"ts": "2026-01-29T10:00:00Z", "watch_id": "test_watch", "query": "retry logic", "hits": [{"chunk_id": "abc123", "score": 0.85, "snippet": "retry with backoff"}]}\n'
    )

    # Setup ingest runs
    runs_log = tmp_path / "ingest_runs.jsonl"
    runs_log.write_text(
        '{"run_id": "run1", "status": "succeeded", "ended_at": "2026-01-29T12:00:00Z", "docs_seen": 10, "docs_changed": 2, "chunks_new": 5, "chunks_updated": 0}\n'
    )

    output_md = tmp_path / "digest.md"
    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=watch_log,
        state_md=None,
        output_md=output_md,
        output_json=None,
        since_hours=24
    )

    content = output_md.read_text()
    assert "Watch Hits" in content
    assert "retry logic" in content

def test_generate_digest_deterministic(tmp_path):
    """Test that digest generation is deterministic."""
    runs_log = tmp_path / "ingest_runs.jsonl"
    runs_log.write_text(
        '{"run_id": "run1", "status": "succeeded", "ended_at": "2026-01-29T12:00:00Z", "docs_seen": 10, "docs_changed": 2, "chunks_new": 5, "chunks_updated": 0}\n'
    )

    output1 = tmp_path / "digest1.md"
    output2 = tmp_path / "digest2.md"

    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=None,
        state_md=None,
        output_md=output1,
        output_json=None,
        since_hours=24
    )

    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=None,
        state_md=None,
        output_md=output2,
        output_json=None,
        since_hours=24
    )

    # Should be identical except for "Generated:" timestamp
    content1 = output1.read_text()
    content2 = output2.read_text()

    # Remove Generated line before comparing
    content1_clean = "\n".join([l for l in content1.split("\n") if not l.startswith("Generated:")])
    content2_clean = "\n".join([l for l in content2.split("\n") if not l.startswith("Generated:")])

    assert content1_clean == content2_clean
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_digest.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `src/chinvex/digest.py`:
```python
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


def generate_digest(
    context_name: str,
    ingest_runs_log: Path | None,
    watch_history_log: Path | None,
    state_md: Path | None,
    output_md: Path,
    output_json: Path | None,
    since_hours: int = 24
) -> None:
    """
    Generate digest markdown and JSON from ingest runs and watch history.
    Deterministic except for "Generated:" timestamp.
    """
    # Calculate since timestamp
    since_ts = datetime.utcnow() - timedelta(hours=since_hours)

    # Gather data
    ingest_stats = _gather_ingest_stats(ingest_runs_log, since_ts)
    watch_hits = _gather_watch_hits(watch_history_log, since_ts)
    state_summary = _extract_state_summary(state_md) if state_md else None

    # Generate markdown
    md_lines = [
        f"# Digest: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
    ]

    if watch_hits:
        md_lines.append(f"## Watch Hits ({len(watch_hits)})")
        for hit in watch_hits:
            md_lines.append(f"- **\"{hit['query']}\"** hit {hit['count']}x")
        md_lines.append("")

    if ingest_stats:
        md_lines.append("## Recent Changes (since last digest)")
        md_lines.append(f"- {ingest_stats['docs_changed']} files ingested, {ingest_stats['chunks_total']} chunks updated")
        md_lines.append("")

    if state_summary:
        md_lines.append("## State Summary")
        md_lines.append(state_summary)
        md_lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines))

    # Generate JSON
    if output_json:
        data = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "context": context_name,
            "since_hours": since_hours,
            "watch_hits": watch_hits,
            "ingest_stats": ingest_stats,
            "state_summary": state_summary
        }
        output_json.write_text(json.dumps(data, indent=2))


def _gather_ingest_stats(log_path: Path | None, since: datetime) -> dict | None:
    """Gather ingest stats since timestamp."""
    if not log_path or not log_path.exists():
        return None

    from .ingest_log import read_ingest_runs
    runs = read_ingest_runs(log_path, completed_only=True)

    # Filter by timestamp
    recent_runs = [
        r for r in runs
        if r["status"] == "succeeded" and
           datetime.fromisoformat(r["ended_at"].replace("Z", "+00:00")) >= since
    ]

    if not recent_runs:
        return None

    # Aggregate
    total_docs_changed = sum(r.get("docs_changed", 0) for r in recent_runs)
    total_chunks_new = sum(r.get("chunks_new", 0) for r in recent_runs)
    total_chunks_updated = sum(r.get("chunks_updated", 0) for r in recent_runs)

    return {
        "docs_changed": total_docs_changed,
        "chunks_total": total_chunks_new + total_chunks_updated,
        "runs": len(recent_runs)
    }


def _gather_watch_hits(log_path: Path | None, since: datetime) -> list[dict]:
    """Gather watch hits since timestamp."""
    if not log_path or not log_path.exists():
        return []

    hits = []
    with log_path.open("r") as f:
        for line in f:
            entry = json.loads(line.strip())
            ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
            if ts >= since:
                hits.append({
                    "query": entry["query"],
                    "count": len(entry["hits"])
                })

    return hits


def _extract_state_summary(state_md: Path) -> str | None:
    """Extract state summary from STATE.md."""
    if not state_md.exists():
        return None

    content = state_md.read_text()
    # Simple extraction: first paragraph under "Current Objective"
    lines = content.split("\n")
    in_objective = False
    summary_lines = []

    for line in lines:
        if line.startswith("## Current Objective"):
            in_objective = True
            continue
        if in_objective:
            if line.startswith("##"):
                break
            if line.strip():
                summary_lines.append(line.strip())

    return " ".join(summary_lines) if summary_lines else None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_digest.py -v`
Expected: PASS (all tests)

**Step 5: Add CLI command**

Modify `src/chinvex/cli.py`:
```python
# Add digest subcommand group
digest_app = typer.Typer(help="Generate digest reports")
app.add_typer(digest_app, name="digest")

@digest_app.command("generate")
def digest_generate_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    since: str = typer.Option("24h", "--since", help="Time window (e.g., 24h, 7d)"),
    date: str | None = typer.Option(None, "--date", help="Generate for specific date (YYYY-MM-DD)"),
    push: str | None = typer.Option(None, "--push", help="Push notification (e.g., ntfy)"),
) -> None:
    """Generate digest for a context."""
    from .context import load_context
    from .context_cli import get_contexts_root
    from .digest import generate_digest
    from pathlib import Path

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Parse since
    import re
    match = re.match(r"(\d+)(h|d)", since)
    if not match:
        typer.secho(f"Invalid --since format: {since}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    amount, unit = match.groups()
    since_hours = int(amount) if unit == "h" else int(amount) * 24

    # Determine output date
    if date:
        from datetime import datetime
        output_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    else:
        from datetime import datetime
        output_date = datetime.now().strftime("%Y-%m-%d")

    # Setup paths
    context_dir = contexts_root / context
    ingest_runs_log = context_dir / "ingest_runs.jsonl"
    watch_history_log = context_dir / "watch_history.jsonl"

    # Find STATE.md (walk up from context to find repo root)
    state_md = None
    search_dir = Path.cwd()
    for _ in range(5):  # Max 5 levels up
        candidate = search_dir / "docs" / "memory" / "STATE.md"
        if candidate.exists():
            state_md = candidate
            break
        search_dir = search_dir.parent

    # Output paths
    digests_dir = context_dir / "digests"
    digests_dir.mkdir(parents=True, exist_ok=True)
    output_md = digests_dir / f"{output_date}.md"
    output_json = digests_dir / f"{output_date}.json"

    generate_digest(
        context_name=context,
        ingest_runs_log=ingest_runs_log,
        watch_history_log=watch_history_log,
        state_md=state_md,
        output_md=output_md,
        output_json=output_json,
        since_hours=since_hours
    )

    typer.secho(f"Digest generated: {output_md}", fg=typer.colors.GREEN)

    # Push notification if requested
    if push == "ntfy":
        _push_ntfy_notification(context, output_md)


def _push_ntfy_notification(context: str, digest_path: Path) -> None:
    """Push notification to ntfy."""
    import os
    import requests

    topic = os.getenv("CHINVEX_NTFY_TOPIC")
    server = os.getenv("CHINVEX_NTFY_SERVER", "https://ntfy.sh")

    if not topic:
        typer.secho("Warning: CHINVEX_NTFY_TOPIC not set, skipping notification", fg=typer.colors.YELLOW)
        return

    # Read digest stats
    content = digest_path.read_text()
    # Extract simple summary
    message = f"Chinvex digest ready for {context}"

    try:
        response = requests.post(
            f"{server}/{topic}",
            data=message,
            headers={"Title": f"Chinvex Digest - {context}"}
        )
        response.raise_for_status()
        typer.secho("Notification sent to ntfy", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to send notification: {e}", fg=typer.colors.RED)
```

**Step 6: Run CLI test**

Test manually:
```bash
chinvex digest generate --context Chinvex --since 24h
```

**Step 7: Commit**

```bash
git add src/chinvex/digest.py src/chinvex/cli.py tests/test_digest.py
git commit -m "feat(digest): add digest generation CLI (P4.3)

Generates deterministic markdown/JSON digests from ingest runs,
watch history, and STATE.md. Supports ntfy notifications.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Operating Brief (P4.4)

### Task 4.1: Brief generation

**Files:**
- Create: `src/chinvex/brief.py`
- Create: `tests/test_brief.py`
- Modify: `src/chinvex/cli.py` (add brief command)

**Step 1: Write the failing test**

Create `tests/test_brief.py`:
```python
import pytest
from pathlib import Path
from chinvex.brief import generate_brief

def test_generate_brief_minimal(tmp_path):
    """Test brief generation with minimal inputs."""
    output = tmp_path / "SESSION_BRIEF.md"

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    assert output.exists()
    content = output.read_text()
    assert "# Session Brief: TestContext" in content
    assert "Generated:" in content

def test_generate_brief_with_state(tmp_path):
    """Test brief includes STATE.md content."""
    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
P4 implementation

## Active Work
- Digest generation
- Brief generation

## Blockers
None

## Next Actions
- [ ] Complete P4.4
""")

    output = tmp_path / "SESSION_BRIEF.md"
    generate_brief(
        context_name="TestContext",
        state_md=state_md,
        constraints_md=None,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()
    assert "Current Objective" in content
    assert "P4 implementation" in content
    assert "Active Work" in content

def test_generate_brief_with_constraints(tmp_path):
    """Test brief includes CONSTRAINTS.md top section."""
    constraints_md = tmp_path / "CONSTRAINTS.md"
    constraints_md.write_text("""# Constraints

## Infrastructure
- ChromaDB batch limit: 5000
- Embedding dims locked

## Rules
- Schema stays v2
""")

    output = tmp_path / "SESSION_BRIEF.md"
    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=constraints_md,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()
    assert "Constraints" in content
    assert "ChromaDB batch limit" in content

def test_generate_brief_with_recent_decisions(tmp_path):
    """Test brief includes recent decisions (last 7 days)."""
    from datetime import datetime, timedelta

    recent_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    decisions_md = tmp_path / "DECISIONS.md"
    decisions_md.write_text(f"""# Decisions

### {recent_date} — Recent decision

- **Why:** Testing
- **Impact:** Should appear
- **Evidence:** commit abc123

### {old_date} — Old decision

- **Why:** Testing
- **Impact:** Should NOT appear
- **Evidence:** commit def456
""")

    output = tmp_path / "SESSION_BRIEF.md"
    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=decisions_md,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()
    assert "Recent decision" in content
    assert "Old decision" not in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `src/chinvex/brief.py`:
```python
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path


def generate_brief(
    context_name: str,
    state_md: Path | None,
    constraints_md: Path | None,
    decisions_md: Path | None,
    latest_digest: Path | None,
    watch_history_log: Path | None,
    output: Path
) -> None:
    """
    Generate session brief from memory files and recent activity.
    Missing files are silently skipped (graceful degradation).
    """
    lines = [
        f"# Session Brief: {context_name}",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]

    # STATE.md: full content
    if state_md and state_md.exists():
        state_content = _extract_state_sections(state_md)
        if state_content:
            lines.extend(state_content)
            lines.append("")

    # CONSTRAINTS.md: top section only (until first ##)
    if constraints_md and constraints_md.exists():
        constraints_content = _extract_constraints_top(constraints_md)
        if constraints_content:
            lines.append("## Constraints (highlights)")
            lines.extend(constraints_content)
            lines.append("")

    # DECISIONS.md: last 7 days
    if decisions_md and decisions_md.exists():
        recent_decisions = _extract_recent_decisions(decisions_md, days=7)
        if recent_decisions:
            lines.append("## Recent Decisions (7d)")
            lines.extend(recent_decisions)
            lines.append("")

    # Latest digest
    if latest_digest and latest_digest.exists():
        digest_summary = _extract_digest_summary(latest_digest)
        if digest_summary:
            lines.append("## Recent Activity")
            lines.extend(digest_summary)
            lines.append("")

    # Watch history: last 5 hits or 24h
    if watch_history_log and watch_history_log.exists():
        watch_summary = _extract_watch_summary(watch_history_log)
        if watch_summary:
            lines.append("## Recent Watch Hits")
            lines.extend(watch_summary)
            lines.append("")

    # Context files reference
    lines.append("## Context Files")
    if state_md:
        lines.append(f"- State: `{state_md}`")
    if latest_digest:
        lines.append(f"- Digest: `{latest_digest}`")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def _extract_state_sections(state_md: Path) -> list[str]:
    """Extract all sections from STATE.md."""
    content = state_md.read_text()
    lines = content.split("\n")

    result = []
    for line in lines:
        if line.startswith("# State"):
            continue  # Skip title
        result.append(line)

    return result


def _extract_constraints_top(constraints_md: Path) -> list[str]:
    """Extract content until first ## heading."""
    content = constraints_md.read_text()
    lines = content.split("\n")

    result = []
    seen_first_section = False

    for line in lines:
        if line.startswith("# Constraints"):
            continue
        if line.startswith("## "):
            if not seen_first_section:
                seen_first_section = True
                result.append(line)
                continue
            else:
                break  # Stop at second ## heading
        result.append(line)

    return result


def _extract_recent_decisions(decisions_md: Path, days: int) -> list[str]:
    """Extract decisions from last N days."""
    content = decisions_md.read_text()
    lines = content.split("\n")

    cutoff_date = datetime.now() - timedelta(days=days)
    result = []
    current_decision = []
    current_date = None

    for line in lines:
        # Match decision heading: ### YYYY-MM-DD — Title
        match = re.match(r"^### (\d{4}-\d{2}-\d{2}) — (.+)", line)
        if match:
            date_str, title = match.groups()
            decision_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Save previous decision if within window
            if current_decision and current_date and current_date >= cutoff_date:
                result.extend(current_decision)
                result.append("")

            # Start new decision
            current_date = decision_date
            current_decision = [line]
        elif current_decision:
            current_decision.append(line)

    # Save last decision
    if current_decision and current_date and current_date >= cutoff_date:
        result.extend(current_decision)

    return result


def _extract_digest_summary(digest_md: Path) -> list[str]:
    """Extract summary from latest digest."""
    content = digest_md.read_text()
    lines = content.split("\n")

    # Extract "Recent Changes" section
    result = []
    in_changes = False

    for line in lines:
        if line.startswith("## Recent Changes"):
            in_changes = True
            continue
        if in_changes:
            if line.startswith("##"):
                break
            result.append(line)

    return result


def _extract_watch_summary(watch_log: Path) -> list[str]:
    """Extract last 5 watch hits or 24h."""
    import json
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(hours=24)
    hits = []

    with watch_log.open("r") as f:
        for line in f:
            entry = json.loads(line.strip())
            ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
            if ts >= cutoff:
                hits.append(entry)

    # Take last 5
    hits = hits[-5:]

    result = []
    for hit in hits:
        result.append(f"- **\"{hit['query']}\"** ({len(hit['hits'])} hits)")

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief.py -v`
Expected: PASS (all tests)

**Step 5: Add CLI command**

Modify `src/chinvex/cli.py`:
```python
@app.command("brief")
def brief_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    output: Path | None = typer.Option(None, "--output", help="Output file (default: stdout)"),
    repo_root: Path | None = typer.Option(None, "--repo-root", help="Repository root (auto-detect if not provided)"),
) -> None:
    """Generate session brief for a context."""
    from .context import load_context
    from .context_cli import get_contexts_root
    from .brief import generate_brief
    from pathlib import Path

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Auto-detect repo root
    if not repo_root:
        repo_root = _find_repo_root(Path.cwd())

    # Setup paths
    state_md = repo_root / "docs" / "memory" / "STATE.md" if repo_root else None
    constraints_md = repo_root / "docs" / "memory" / "CONSTRAINTS.md" if repo_root else None
    decisions_md = repo_root / "docs" / "memory" / "DECISIONS.md" if repo_root else None

    context_dir = contexts_root / context
    digests_dir = context_dir / "digests"
    latest_digest = _find_latest_digest(digests_dir)
    watch_history_log = context_dir / "watch_history.jsonl"

    # Generate brief
    if output:
        output_path = output
    else:
        import tempfile
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        output_path = Path(temp.name)

    generate_brief(
        context_name=context,
        state_md=state_md,
        constraints_md=constraints_md,
        decisions_md=decisions_md,
        latest_digest=latest_digest,
        watch_history_log=watch_history_log,
        output=output_path
    )

    if output:
        typer.secho(f"Brief generated: {output_path}", fg=typer.colors.GREEN)
    else:
        # Print to stdout
        content = output_path.read_text()
        typer.echo(content)
        output_path.unlink()  # Clean up temp file


def _find_repo_root(start_dir: Path) -> Path | None:
    """Walk up to find repo root (has .git or docs/memory)."""
    current = start_dir
    for _ in range(5):  # Max 5 levels
        if (current / ".git").exists() or (current / "docs" / "memory").exists():
            return current
        current = current.parent
    return None


def _find_latest_digest(digests_dir: Path) -> Path | None:
    """Find most recent digest by filename (YYYY-MM-DD.md)."""
    if not digests_dir.exists():
        return None

    digests = sorted(digests_dir.glob("*.md"), reverse=True)
    return digests[0] if digests else None
```

**Step 6: Run CLI test**

Test manually:
```bash
chinvex brief --context Chinvex
```

**Step 7: Commit**

```bash
git add src/chinvex/brief.py src/chinvex/cli.py tests/test_brief.py
git commit -m "feat(brief): add session brief generation (P4.4)

Generates brief from STATE.md, CONSTRAINTS.md, DECISIONS.md,
latest digest, and watch history. Graceful degradation if files missing.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Startup Hook (P4.5)

### Task 5.1: Update CLAUDE.md with session protocol

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add session start protocol**

Modify `CLAUDE.md`, add section:
```markdown
## Session Start Protocol

On session start, run: `chinvex brief --context Chinvex`

Read the output before proceeding with any work. This loads current state,
constraints, recent decisions, and activity into your context.

**Why:** Ensures Claude starts with awareness of project state rather than
guessing or relying on stale information.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add session start protocol to CLAUDE.md (P4.5)

Instructs Claude to run chinvex brief on session start to load
current state and context before beginning work.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: Observability (P4.6)

### Task 6.1: Add embedding metrics

**Files:**
- Modify: `src/chinvex/gateway/metrics.py`
- Modify: `src/chinvex/embedding_providers.py` (add metrics)
- Create: `tests/test_embedding_metrics.py`

**Step 1: Write the failing test**

Create `tests/test_embedding_metrics.py`:
```python
import pytest
from chinvex.gateway.metrics import get_metrics_registry
from chinvex.embedding_providers import OllamaProvider

def test_embedding_metrics_tracked():
    """Test that embedding calls are tracked in metrics."""
    registry = get_metrics_registry()

    provider = OllamaProvider("http://localhost:11434", "mxbai-embed-large")

    # Mock embed call
    with pytest.raises(Exception):  # Will fail due to no Ollama server
        provider.embed(["test"])

    # Check metrics exist
    metrics_text = registry.collect()
    # Should have chinvex_embeddings_total{provider="ollama"}
```

**Step 2: Add metrics to providers**

Modify `src/chinvex/embedding_providers.py`:
```python
from prometheus_client import Counter, Histogram

# Metrics
EMBEDDINGS_TOTAL = Counter(
    "chinvex_embeddings_total",
    "Total embedding requests",
    ["provider"]
)

EMBEDDINGS_LATENCY = Histogram(
    "chinvex_embeddings_latency_seconds",
    "Embedding request latency",
    ["provider"]
)

EMBEDDINGS_RETRIES = Counter(
    "chinvex_embeddings_retries_total",
    "Total embedding retries",
    ["provider"]
)


class OllamaProvider:
    # ... existing code ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        EMBEDDINGS_TOTAL.labels(provider="ollama").inc()

        with EMBEDDINGS_LATENCY.labels(provider="ollama").time():
            return self._embed_impl(texts)


class OpenAIProvider:
    # ... existing code ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        EMBEDDINGS_TOTAL.labels(provider="openai").inc()

        with EMBEDDINGS_LATENCY.labels(provider="openai").time():
            all_embeddings = []
            for i in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = texts[i:i + self.MAX_BATCH_SIZE]
                embeddings = self._embed_batch(batch)
                all_embeddings.extend(embeddings)
            return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(...)
                return [item.embedding for item in response.data]
            except RateLimitError:
                EMBEDDINGS_RETRIES.labels(provider="openai").inc()
                # ... retry logic ...
```

**Step 3: Add digest/brief metrics**

Modify `src/chinvex/digest.py` and `src/chinvex/brief.py`:
```python
from prometheus_client import Counter

DIGEST_GENERATED = Counter(
    "chinvex_digest_generated_total",
    "Total digests generated",
    ["context"]
)

def generate_digest(...):
    DIGEST_GENERATED.labels(context=context_name).inc()
    # ... existing logic ...
```

Similar for brief.

**Step 4: Commit**

```bash
git add src/chinvex/embedding_providers.py src/chinvex/digest.py src/chinvex/brief.py tests/test_embedding_metrics.py
git commit -m "feat(metrics): add embedding and digest metrics (P4.6)

Tracks embedding requests, latency, retries per provider.
Tracks digest and brief generation per context.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 7: Runbooks (P4.7)

### Task 7.1: Create runbook scripts

**Files:**
- Create: `scripts/start_mcp.ps1`
- Create: `scripts/backup.ps1`

**Step 1: Create start_mcp.ps1**

Create `scripts/start_mcp.ps1`:
```powershell
# Start MCP server with token from secrets
$env:CHINVEX_API_TOKEN = Get-Content ~/.secrets/chinvex_token
chinvex-mcp
```

**Step 2: Create backup.ps1**

Create `scripts/backup.ps1`:
```powershell
# Snapshot context registry + indexes + digests
$timestamp = Get-Date -Format "yyyy-MM-dd"
$dest = "P:\backups\chinvex\$timestamp"

New-Item -ItemType Directory -Force -Path $dest

Copy-Item -Recurse P:\ai_memory\contexts "$dest\contexts"
Copy-Item -Recurse P:\ai_memory\indexes "$dest\indexes"

Write-Host "Backup complete: $dest"
```

**Step 3: Commit**

```bash
git add scripts/start_mcp.ps1 scripts/backup.ps1
git commit -m "feat(ops): add runbook scripts (P4.7)

Added start_mcp.ps1 and backup.ps1 for common operations.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 8: End-to-End Testing (P4.8)

### Task 8.1: E2E smoke test

**Files:**
- Create: `scripts/e2e_smoke_p4.py`

**Step 1: Write E2E smoke test**

Create `scripts/e2e_smoke_p4.py`:
```python
#!/usr/bin/env python3
"""
P4 E2E smoke test.

Verifies:
- Context creation with --repo
- OpenAI embedding provider (mocked)
- Digest generation
- Brief generation
"""

import tempfile
import shutil
from pathlib import Path
import subprocess

def run(cmd: list[str]):
    """Run command and return output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}")
        print(result.stdout)
        print(result.stderr)
        exit(1)
    return result.stdout

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Setup test repo
        test_repo = tmp / "test_repo"
        test_repo.mkdir()
        (test_repo / "test.py").write_text("def hello(): pass")

        # Setup contexts root
        contexts_root = tmp / "contexts"
        contexts_root.mkdir()

        print("✓ Test environment setup")

        # Test: Create context with --repo
        run([
            "chinvex", "ingest",
            "--context", "TestP4",
            "--repo", str(test_repo),
        ])

        assert (contexts_root / "TestP4" / "context.json").exists()
        print("✓ Context creation with --repo")

        # Test: Digest generation
        run([
            "chinvex", "digest", "generate",
            "--context", "TestP4",
            "--since", "24h"
        ])

        digests_dir = contexts_root / "TestP4" / "digests"
        assert any(digests_dir.glob("*.md"))
        print("✓ Digest generation")

        # Test: Brief generation
        output = run([
            "chinvex", "brief",
            "--context", "TestP4"
        ])

        assert "Session Brief: TestP4" in output
        print("✓ Brief generation")

        print("\n✓ All P4 smoke tests passed!")

if __name__ == "__main__":
    main()
```

**Step 2: Run smoke test**

```bash
python scripts/e2e_smoke_p4.py
```

**Step 3: Commit**

```bash
git add scripts/e2e_smoke_p4.py
git commit -m "test(e2e): add P4 smoke test (P4.8)

Validates context creation, digest, and brief generation.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Final Steps

### Task 9.1: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add P4 features to README**

Add section after "MCP Server":
```markdown
## Digest & Brief

Generate daily digests and session briefs from your indexed content.

### Digest

```powershell
# Generate digest for last 24 hours
chinvex digest generate --context MyProject --since 24h

# Generate for specific date
chinvex digest generate --context MyProject --date 2026-01-28

# Push notification to ntfy
chinvex digest generate --context MyProject --push ntfy
```

### Brief

```powershell
# Generate session brief
chinvex brief --context MyProject

# Save to file
chinvex brief --context MyProject --output SESSION_BRIEF.md
```

### Memory Files

Create `docs/memory/` in your repo:
- `STATE.md`: Current objective, active work, blockers
- `CONSTRAINTS.md`: Infrastructure limits, rules, key facts
- `DECISIONS.md`: Append-only decision log

See [Memory File Format](specs/P4_IMPLEMENTATION_SPEC.md#appendix-memory-file-format) for details.

## Embedding Providers

Chinvex supports multiple embedding providers:

```powershell
# Use Ollama (default)
chinvex ingest --context MyProject

# Use OpenAI
chinvex ingest --context MyProject --embed-provider openai

# Switch providers (requires rebuild)
chinvex ingest --context MyProject --embed-provider openai --rebuild-index
```

Set `OPENAI_API_KEY` environment variable for OpenAI provider.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add P4 features to README

Documents digest, brief, memory files, and embedding providers.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9.2: Update DECISIONS.md

**Files:**
- Modify: `docs/memory/DECISIONS.md`

**Step 1: Add P4 decision entry**

Append to `docs/memory/DECISIONS.md`:
```markdown
### 2026-01-29 — P4 implementation complete

- **Why:** Session bootstrap + daily digest needed for daily-usable memory service
- **Impact:** Claude starts with context loaded, proactive surfacing via digest/brief
- **Evidence:** `specs/P4_IMPLEMENTATION_SPEC.md`, `docs/plans/2026-01-29-p4-implementation.md`
```

**Step 2: Commit**

```bash
git add docs/memory/DECISIONS.md
git commit -m "docs: record P4 implementation decision

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Execution Complete

All tasks implemented. Ready to test end-to-end.

**Recommended validation:**
1. Run full test suite: `pytest tests/ -v`
2. Run E2E smoke test: `python scripts/e2e_smoke_p4.py`
3. Manual test: Create new context, ingest, generate digest, generate brief
4. Verify metrics endpoint exposes new metrics

**Next steps:**
- Deploy to production
- Monitor OpenAI costs (if using openai provider)
- Set up daily digest cron job (future P5)
