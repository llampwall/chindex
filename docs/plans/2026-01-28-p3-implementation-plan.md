# P3 Quality of Life Release Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver quality-of-life improvements: smarter chunking, cross-context search, proactive alerts, and archive tier.

**Architecture:** Incremental enhancements across chunking, search, CLI, and gateway layers. Cross-context search reuses existing indexes. Chunking v2 requires re-index. Archive tier uses flag-based filtering.

**Tech Stack:** Python AST, Redis (optional), Prometheus client, existing Chinvex core

---

## Prerequisites

Before starting implementation:

1. P2 must be complete
2. Python 3.12+ environment
3. Existing Chinvex installation working
4. Test contexts with ingested data
5. Gateway running and tested

---

## P3a: Quality + Convenience

### Phase 1: Cross-Context Search (P3.3)

Priority: Ship first — works with existing indexes, immediate user value

---

### Task 1: Multi-context search core logic

**Files:**
- Create: `tests/test_cross_context_search.py`
- Modify: `src/chinvex/search.py`

**Step 1: Write failing test for multi-context search**

Create `tests/test_cross_context_search.py`:

```python
"""Test cross-context search functionality."""
import pytest
from chinvex.search import search_multi_context, SearchResult
from chinvex.context import Context


def test_search_multi_context_merges_results(tmp_path):
    """Test that search_multi_context merges results by score."""
    # Setup: Create two contexts with test data
    ctx1_dir = tmp_path / "ctx1"
    ctx2_dir = tmp_path / "ctx2"
    ctx1_dir.mkdir()
    ctx2_dir.mkdir()

    # Create minimal contexts (implementation will come later)
    ctx1 = Context(name="Context1", base_dir=str(ctx1_dir))
    ctx2 = Context(name="Context2", base_dir=str(ctx2_dir))

    # Mock search results
    results = search_multi_context(
        contexts=["Context1", "Context2"],
        query="test",
        k=5
    )

    # Expected: Returns list of SearchResult objects
    assert isinstance(results, list)
    # Expected: Results are tagged with context name
    for r in results:
        assert hasattr(r, 'context')
        assert r.context in ["Context1", "Context2"]


def test_search_multi_context_respects_k_limit(tmp_path):
    """Test that k limits total results, not per-context."""
    results = search_multi_context(
        contexts=["Context1", "Context2"],
        query="test",
        k=10
    )
    assert len(results) <= 10


def test_search_multi_context_sorts_by_score(tmp_path):
    """Test that results are sorted by score descending."""
    results = search_multi_context(
        contexts=["Context1", "Context2"],
        query="test",
        k=10
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_multi_context_all_expands_to_all_contexts(tmp_path):
    """Test that 'all' expands to all available contexts."""
    results = search_multi_context(
        contexts="all",
        query="test",
        k=10
    )
    assert isinstance(results, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cross_context_search.py -v`
Expected: FAIL with "function not defined" or import error

**Step 3: Add search_multi_context function**

Edit `src/chinvex/search.py`, add at end:

```python
def search_multi_context(
    contexts: list[str] | str,
    query: str,
    k: int = 10,
    ollama_host: str = "http://localhost:11434",
    include_archive: bool = False,
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.

    Args:
        contexts: List of context names, or "all" for all contexts
        query: Search query
        k: Total number of results to return (not per-context)
        ollama_host: Ollama host for embedding
        include_archive: Include archived documents

    Returns:
        List of SearchResult objects sorted by score descending
    """
    from chinvex.context import list_contexts, load_context

    # Expand "all" to all available contexts
    if contexts == "all":
        contexts = [c.name for c in list_contexts()]

    # Cap contexts to prevent slowdown
    max_contexts = 10  # TODO: Make configurable
    if len(contexts) > max_contexts:
        contexts = contexts[:max_contexts]

    # Per-context cap: fetch more than k to ensure good merged results
    k_per_context = min(k * 2, 20)

    # Gather results from each context
    all_results = []
    for ctx_name in contexts:
        try:
            ctx = load_context(ctx_name)
            results = search_hybrid(
                ctx=ctx,
                query=query,
                k=k_per_context,
                ollama_host=ollama_host,
                source_filter=None,
                include_archive=include_archive,
            )
            # Tag each result with source context
            for r in results:
                r.context = ctx_name
            all_results.extend(results)
        except Exception as e:
            # Log warning but continue with other contexts
            print(f"Warning: Failed to search context {ctx_name}: {e}")
            continue

    # Sort by score descending, take top k
    all_results.sort(key=lambda r: r.score, reverse=True)
    final_results = all_results[:k]

    # Debug logging: score distribution across contexts
    if final_results:
        score_min = min(r.score for r in final_results)
        score_max = max(r.score for r in final_results)
        context_counts = {}
        for r in final_results:
            context_counts[r.context] = context_counts.get(r.context, 0) + 1
        print(f"[DEBUG] Cross-context scores: min={score_min:.3f}, max={score_max:.3f}, by_context={context_counts}")

    return final_results
```

**Step 4: Add context field to SearchResult**

Edit `src/chinvex/search.py`, find SearchResult dataclass and add field:

```python
@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source_type: str
    source_uri: str
    updated_at: str | None = None
    context: str | None = None  # NEW: Source context for multi-context search
```

**Step 5: Add list_contexts helper**

Edit `src/chinvex/context.py`, add at end:

```python
def list_contexts(base_dir: str = "P:/ai_memory/contexts") -> list[Context]:
    """List all available contexts."""
    from pathlib import Path

    contexts = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return contexts

    for ctx_dir in base_path.iterdir():
        if ctx_dir.is_dir() and (ctx_dir / "context.json").exists():
            try:
                ctx = load_context(ctx_dir.name, base_dir=str(base_dir))
                contexts.append(ctx)
            except Exception:
                continue

    return contexts
```

**Step 6: Run tests**

Run: `pytest tests/test_cross_context_search.py -v`
Expected: Tests pass (or reveal integration issues to fix)

**Step 7: Commit**

```bash
git add src/chinvex/search.py src/chinvex/context.py tests/test_cross_context_search.py
git commit -m "feat(search): add multi-context search support

- Add search_multi_context() for cross-context queries
- Tag results with source context
- Support 'all' keyword for all contexts
- Merge results by score with configurable k limit

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: CLI cross-context search commands

**Files:**
- Modify: `src/chinvex/cli.py`
- Create: `tests/test_cli_cross_context.py`

**Step 1: Write failing test**

Create `tests/test_cli_cross_context.py`:

```python
"""Test CLI cross-context search commands."""
import pytest
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_search_all_flag():
    """Test chinvex search --all flag."""
    result = runner.invoke(app, ["search", "--all", "test query"])
    assert result.exit_code == 0
    assert "Context:" in result.stdout or "Searched contexts:" in result.stdout


def test_search_contexts_flag():
    """Test chinvex search --contexts flag."""
    result = runner.invoke(app, ["search", "--contexts", "Chinvex,Personal", "test query"])
    assert result.exit_code == 0


def test_search_exclude_flag():
    """Test chinvex search --exclude flag."""
    result = runner.invoke(app, ["search", "--all", "--exclude", "Work", "test query"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_cross_context.py -v`
Expected: FAIL with "no such option: --all"

**Step 3: Add CLI flags to search command**

Edit `src/chinvex/cli.py`, find the `search` command and modify:

```python
@app.command()
def search(
    query: str,
    context: str | None = Option(None, "--context", "-c", help="Single context name (deprecated, use --contexts)"),
    contexts: str | None = Option(None, "--contexts", help="Comma-separated context names"),
    all_contexts: bool = Option(False, "--all", help="Search all contexts"),
    exclude: str | None = Option(None, "--exclude", help="Comma-separated contexts to exclude (with --all)"),
    source: str | None = Option(None, "--source", help="Filter by source type (repo/chat)"),
    k: int = Option(10, "--k", "-k", help="Number of results"),
    ollama_host: str = Option("http://localhost:11434", "--ollama-host", help="Ollama API host"),
    include_archive: bool = Option(False, "--include-archive", help="Include archived documents"),
):
    """Search for content across context(s)."""
    from chinvex.search import search_hybrid, search_multi_context
    from chinvex.context import load_context

    # Determine which contexts to search
    if all_contexts:
        ctx_list = "all"
        if exclude:
            # Filter out excluded contexts
            from chinvex.context import list_contexts
            all_ctx = [c.name for c in list_contexts()]
            excluded = [x.strip() for x in exclude.split(",")]
            ctx_list = [c for c in all_ctx if c not in excluded]
    elif contexts:
        ctx_list = [x.strip() for x in contexts.split(",")]
    elif context:
        # Legacy single-context mode
        ctx = load_context(context)
        results = search_hybrid(
            ctx=ctx,
            query=query,
            k=k,
            ollama_host=ollama_host,
            source_filter=source,
            include_archive=include_archive,
        )
        _print_search_results(results)
        return
    else:
        print("Error: Must specify --context, --contexts, or --all")
        raise typer.Exit(1)

    # Multi-context search
    results = search_multi_context(
        contexts=ctx_list,
        query=query,
        k=k,
        ollama_host=ollama_host,
        include_archive=include_archive,
    )

    # Print results with context tags
    if not results:
        print(f"No results for: {query}")
        return

    print(f"\nSearched contexts: {ctx_list if isinstance(ctx_list, list) else 'all'}")
    print(f"Found {len(results)} results")

    # Score distribution stats (for detecting cross-context anomalies)
    if results:
        score_min = min(r.score for r in results)
        score_max = max(r.score for r in results)
        context_counts = {}
        for r in results:
            context_counts[r.context] = context_counts.get(r.context, 0) + 1
        print(f"Score range: {score_min:.3f} - {score_max:.3f}")
        print(f"Results by context: {context_counts}\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [Context: {r.context}] [{r.source_type}] {r.source_uri}")
        print(f"   Score: {r.score:.3f}")
        print(f"   {r.text[:150]}...")
        print()
```

**Step 4: Run tests**

Run: `pytest tests/test_cli_cross_context.py -v`
Expected: Tests pass

**Step 5: Test manually**

Run: `chinvex search --all "test" --k 5`
Expected: Results from multiple contexts displayed

**Step 6: Commit**

```bash
git add src/chinvex/cli.py tests/test_cli_cross_context.py
git commit -m "feat(cli): add cross-context search flags

- Add --all flag to search all contexts
- Add --contexts flag for specific contexts
- Add --exclude flag to exclude contexts
- Maintain backward compatibility with --context

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Gateway API multi-context support

**Files:**
- Modify: `src/chinvex/gateway/endpoints/search.py`
- Modify: `src/chinvex/gateway/endpoints/evidence.py`
- Create: `tests/gateway/test_multi_context_api.py`

**Step 1: Write failing API test**

Create `tests/gateway/test_multi_context_api.py`:

```python
"""Test gateway multi-context API endpoints."""
import pytest
from fastapi.testclient import TestClient
from chinvex.gateway.app import app

client = TestClient(app)


def test_search_accepts_contexts_array():
    """Test /v1/search accepts contexts array."""
    response = client.post(
        "/v1/search",
        json={
            "contexts": ["Chinvex", "Personal"],
            "query": "test",
            "k": 5
        },
        headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "contexts_searched" in data
    assert "results" in data


def test_search_accepts_contexts_all():
    """Test /v1/search accepts contexts='all'."""
    response = client.post(
        "/v1/search",
        json={
            "contexts": "all",
            "query": "test",
            "k": 5
        },
        headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200


def test_search_backward_compat_context_singular():
    """Test /v1/search still accepts context (singular)."""
    response = client.post(
        "/v1/search",
        json={
            "context": "Chinvex",
            "query": "test",
            "k": 5
        },
        headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200


def test_contexts_takes_precedence_over_context():
    """Test contexts (plural) takes precedence if both provided."""
    response = client.post(
        "/v1/search",
        json={
            "context": "Chinvex",
            "contexts": ["Personal"],  # This should win
            "query": "test"
        },
        headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["contexts_searched"] == ["Personal"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/test_multi_context_api.py -v`
Expected: FAIL with validation error or 422

**Step 3: Update search endpoint schema**

Edit `src/chinvex/gateway/endpoints/search.py`:

```python
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request schema with multi-context support."""
    context: str | None = Field(None, description="Single context (deprecated, use contexts)")
    contexts: list[str] | str | None = Field(None, description="Context names or 'all'")
    query: str = Field(..., description="Search query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    source: str | None = Field(None, description="Filter by source type")
    include_archive: bool = Field(False, description="Include archived documents")
    debug: bool = Field(False, description="Include debug fields (score distribution, context counts)")


class SearchResponse(BaseModel):
    """
    Search response schema.

    Debug fields (score_min, score_max, results_by_context) are only populated when debug=true.
    This prevents leaking corpus shape to ChatGPT Actions.
    """
    query: str
    contexts_searched: list[str] | str
    results: list[dict]
    total_results: int
    # Debug-only fields (opt-in via debug=true)
    score_min: float | None = None
    score_max: float | None = None
    results_by_context: dict[str, int] | None = None


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest, ctx: dict = Depends(get_request_context)):
    """
    Search across one or more contexts.

    Supports both single-context (legacy) and multi-context search.
    """
    from chinvex.search import search_hybrid, search_multi_context
    from chinvex.context import load_context
    from chinvex.gateway.config import get_gateway_config

    config = get_gateway_config()

    # Determine contexts to search
    if req.contexts:
        # Multi-context mode (new)
        contexts = req.contexts
    elif req.context:
        # Single-context mode (legacy)
        contexts = [req.context]
    else:
        return {"error": "Must specify context or contexts"}, 400

    # Apply allowlist filtering
    if config.context_allowlist:
        if isinstance(contexts, str) and contexts == "all":
            contexts = config.context_allowlist
        else:
            # Filter contexts against allowlist
            allowed = set(config.context_allowlist)
            filtered = [c for c in contexts if c in allowed]
            # Log warning for excluded contexts
            excluded = set(contexts) - set(filtered)
            if excluded:
                for ctx_name in excluded:
                    print(f"Warning: Context '{ctx_name}' not in allowlist, skipping")
            contexts = filtered

    # Perform search
    if isinstance(contexts, list) and len(contexts) == 1:
        # Single context (optimize for legacy path)
        ctx = load_context(contexts[0])
        results = search_hybrid(
            ctx=ctx,
            query=req.query,
            k=req.k,
            source_filter=req.source,
            include_archive=req.include_archive,
        )
        results_dicts = [r.to_dict() for r in results]
        return SearchResponse(
            query=req.query,
            contexts_searched=contexts,
            results=results_dicts,
            total_results=len(results_dicts),
        )
    else:
        # Multi-context search
        results = search_multi_context(
            contexts=contexts,
            query=req.query,
            k=req.k,
            include_archive=req.include_archive,
        )
        results_dicts = [r.to_dict() for r in results]

        # Build response (conditionally include debug fields)
        response_data = {
            "query": req.query,
            "contexts_searched": contexts,
            "results": results_dicts,
            "total_results": len(results_dicts),
        }

        # Only add debug fields if explicitly requested (don't leak to ChatGPT Actions)
        if req.debug and results:
            score_min = min(r.score for r in results)
            score_max = max(r.score for r in results)
            results_by_context = {}
            for r in results:
                results_by_context[r.context] = results_by_context.get(r.context, 0) + 1

            response_data["score_min"] = score_min
            response_data["score_max"] = score_max
            response_data["results_by_context"] = results_by_context

        return SearchResponse(**response_data)
```

**Step 4: Add to_dict method to SearchResult**

Edit `src/chinvex/search.py`, add method to SearchResult:

```python
@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source_type: str
    source_uri: str
    updated_at: str | None = None
    context: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "updated_at": self.updated_at,
            "context": self.context,
        }
```

**Step 5: Update evidence endpoint similarly**

Edit `src/chinvex/gateway/endpoints/evidence.py`, apply same pattern as search.

**Step 6: Run tests**

Run: `pytest tests/gateway/test_multi_context_api.py -v`
Expected: Tests pass

**Step 7: Commit**

```bash
git add src/chinvex/gateway/endpoints/search.py src/chinvex/gateway/endpoints/evidence.py src/chinvex/search.py tests/gateway/test_multi_context_api.py
git commit -m "feat(gateway): add multi-context search API support

- Accept both context and contexts parameters
- Support contexts='all' for all contexts
- Apply allowlist filtering with silent exclusion
- Maintain backward compatibility with P2 clients

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Phase 2: Chunking v2 (P3.1)

Priority: Requires re-index, but improves all future queries

---

### Task 4: Overlap for generic chunking

**Files:**
- Create: `tests/test_chunking_v2.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Write failing test**

Create `tests/test_chunking_v2.py`:

```python
"""Test chunking v2 improvements."""
import pytest
from chinvex.chunking import chunk_with_overlap, chunk_generic_file


def test_chunk_with_overlap_basic():
    """Test that overlap is applied between chunks."""
    text = "a" * 5000  # 5000 chars
    chunks = chunk_with_overlap(text, size=3000, overlap=300)

    # Expected: 2-3 chunks with overlap
    assert len(chunks) >= 2

    # Check overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        start1, end1 = chunks[i]
        start2, end2 = chunks[i + 1]

        # start2 should be before end1 (overlap)
        assert start2 < end1
        # Overlap should be ~300 chars
        overlap_size = end1 - start2
        assert 250 <= overlap_size <= 350


def test_chunk_with_overlap_handles_small_text():
    """Test that small text becomes single chunk."""
    text = "small text"
    chunks = chunk_with_overlap(text, size=3000, overlap=300)
    assert len(chunks) == 1
    assert chunks[0] == (0, len(text))


def test_chunk_generic_file_uses_overlap():
    """Test that generic file chunking uses overlap."""
    text = "Lorem ipsum " * 500  # ~6000 chars
    chunks = chunk_generic_file(text)

    assert len(chunks) >= 2
    # Verify chunks have overlap metadata
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        # Check that text overlaps
        text1 = text[chunk1.char_start:chunk1.char_end]
        text2 = text[chunk2.char_start:chunk2.char_end]
        # Last ~300 chars of chunk1 should appear in chunk2
        overlap_text = text1[-300:]
        assert overlap_text in text2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_v2.py::test_chunk_with_overlap_basic -v`
Expected: FAIL with "function not defined"

**Step 3: Implement chunk_with_overlap**

Edit `src/chinvex/chunking.py`, add at end:

```python
def chunk_with_overlap(text: str, size: int = 3000, overlap: int = 300) -> list[tuple[int, int]]:
    """
    Return list of (start, end) positions for chunks with overlap.

    Generic fallback for prose and unknown file types.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_generic_file(text: str, size: int = 3000, overlap: int = 300) -> list[Chunk]:
    """
    Chunk generic text file with overlap.

    Used for: txt, unknown extensions, prose files.
    """
    positions = chunk_with_overlap(text, size, overlap)
    chunks = []
    for ordinal, (start, end) in enumerate(positions):
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            ordinal=ordinal,
            char_start=start,
            char_end=end,
        ))
    return chunks
```

**Step 4: Run tests**

Run: `pytest tests/test_chunking_v2.py::test_chunk_with_overlap_basic -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_chunking_v2.py
git commit -m "feat(chunking): add overlap support for generic chunking

- Implement chunk_with_overlap() for prose
- Add chunk_generic_file() using overlap
- Overlap defaults to 300 chars (~10%)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Semantic boundary detection

**Files:**
- Modify: `tests/test_chunking_v2.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Write failing test**

Add to `tests/test_chunking_v2.py`:

```python
def test_find_best_split_prefers_headers():
    """Test that semantic boundaries are preferred."""
    from chinvex.chunking import find_best_split

    text = "a" * 2800 + "\n## Header\n" + "b" * 500
    # Target split at 3000, should prefer header at 2801
    split_pos = find_best_split(text, target_pos=3000, size=3000)

    # Should split at or near the header
    assert 2700 <= split_pos <= 2900
    assert text[split_pos:split_pos + 10].startswith("\n## ")


def test_find_best_split_handles_no_boundaries():
    """Test fallback when no semantic boundaries found."""
    from chinvex.chunking import find_best_split

    text = "a" * 5000  # No boundaries
    split_pos = find_best_split(text, target_pos=3000, size=3000)

    # Should return near target (may find \n or target itself)
    assert 2700 <= split_pos <= 3300


def test_chunk_markdown_respects_boundaries():
    """Test that markdown files chunk at headers."""
    from chinvex.chunking import chunk_markdown_file

    text = """# Title

Some intro text.

## Section 1

""" + ("Content " * 500) + """

## Section 2

""" + ("More content " * 500)

    chunks = chunk_markdown_file(text)

    # Verify chunks start at or near headers
    for chunk in chunks:
        if chunk.ordinal > 0:  # Skip first chunk
            chunk_text = text[chunk.char_start:chunk.char_end]
            # Should start near a header or section boundary
            assert chunk_text.startswith(("\n", "#", "##")) or \
                   text[max(0, chunk.char_start - 10):chunk.char_start].count("\n") >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_v2.py::test_find_best_split_prefers_headers -v`
Expected: FAIL with "function not defined"

**Step 3: Implement semantic boundary detection**

Edit `src/chinvex/chunking.py`, add:

```python
import re

# Semantic boundary priorities (pattern, score)
SPLIT_PRIORITIES = [
    (r'\n## ', 100),           # Markdown H2
    (r'\n### ', 90),           # Markdown H3
    (r'\n---\n', 85),          # Markdown horizontal rule
    (r'\nclass ', 80),         # Python class (heuristic)
    (r'\ndef ', 75),           # Python function (heuristic)
    (r'\nasync def ', 75),     # Python async function
    (r'\nfunction ', 75),      # JS function (heuristic)
    (r'\nasync function ', 75),# JS async function
    (r'\nconst \w+ = \(', 72), # JS arrow function
    (r'\nconst \w+ = async \(', 72),  # JS async arrow
    (r'\nconst \w+ = ', 70),   # JS const declaration
    (r'\nexport default ', 70),# JS/TS default export
    (r'\nexport ', 68),        # JS/TS named export
    (r'\nmodule\.exports', 68),# CommonJS export
    (r'\n\n\n', 60),           # Multiple blank lines
    (r'\n\n', 50),             # Paragraph break
    (r'\n', 10),               # Line break (last resort)
]


def find_best_split(text: str, target_pos: int, size: int = 3000) -> int:
    """
    Find best split point near target_pos.

    Searches within ±window chars for highest-priority boundary.
    Returns position of the boundary (start of the pattern match).
    """
    window = min(800, max(300, size // 10))  # Proportional window
    search_start = max(0, target_pos - window)
    search_end = min(len(text), target_pos + window)
    search_region = text[search_start:search_end]

    best_pos = target_pos
    best_priority = 0

    for pattern, priority in SPLIT_PRIORITIES:
        for match in re.finditer(pattern, search_region):
            match_pos = search_start + match.start()
            # Prefer boundaries closer to target
            distance_penalty = abs(match_pos - target_pos) / window * 10
            effective_priority = priority - distance_penalty
            if effective_priority > best_priority:
                best_priority = effective_priority
                best_pos = match_pos

    return best_pos


def chunk_markdown_file(text: str, max_chars: int = 3000) -> list[Chunk]:
    """
    Chunk markdown file respecting headers and section boundaries.

    IMPORTANT: NO overlap between chunks. Semantic boundaries define splits.
    """
    chunks = []
    start = 0
    ordinal = 0

    while start < len(text):
        # Target end position
        target_end = start + max_chars

        if target_end >= len(text):
            # Last chunk
            chunks.append(Chunk(
                text=text[start:],
                ordinal=ordinal,
                char_start=start,
                char_end=len(text),
            ))
            break

        # Find best split point near target (uses semantic boundaries)
        split_pos = find_best_split(text, target_end, max_chars)

        chunks.append(Chunk(
            text=text[start:split_pos],
            ordinal=ordinal,
            char_start=start,
            char_end=split_pos,
        ))

        ordinal += 1
        start = split_pos  # NO overlap - next chunk starts at boundary

    return chunks
```

**Step 4: Run tests**

Run: `pytest tests/test_chunking_v2.py -k "semantic" -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_chunking_v2.py
git commit -m "feat(chunking): add semantic boundary detection

- Implement find_best_split() with priority-based boundaries
- Add markdown-aware chunking (respects headers)
- Proportional search window for large code blocks

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Python code-aware splitting (AST)

**Files:**
- Modify: `tests/test_chunking_v2.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Write failing test**

Add to `tests/test_chunking_v2.py`:

```python
def test_extract_python_boundaries():
    """Test Python AST boundary extraction."""
    from chinvex.chunking import extract_python_boundaries

    text = '''"""Module docstring."""

def function_one():
    return 1

class MyClass:
    def method(self):
        pass

@decorator
def function_two():
    return 2

if __name__ == "__main__":
    main()
'''

    boundaries = extract_python_boundaries(text)

    # Expected: boundaries at function/class definitions
    assert len(boundaries) >= 3  # function_one, MyClass, function_two, __main__
    # First boundary should be after docstring
    assert boundaries[0] > 20  # After module docstring


def test_chunk_python_file_respects_functions():
    """Test that Python files chunk at function boundaries."""
    from chinvex.chunking import chunk_python_file

    # Create Python file with multiple functions
    functions = []
    for i in range(10):
        functions.append(f'''
def function_{i}():
    """Function {i} docstring."""
    # Implementation
    {f"x = {i}" * 50}
    return {i}
''')

    text = "\n".join(functions)
    chunks = chunk_python_file(text, max_chars=3000)

    # Verify multiple chunks created
    assert len(chunks) >= 2

    # Verify each chunk starts at a function boundary (or start of file)
    for chunk in chunks:
        if chunk.ordinal > 0:
            chunk_text = text[chunk.char_start:chunk.char_end]
            # Should start with 'def ' or at file start
            assert chunk_text.lstrip().startswith("def ") or chunk.char_start == 0


def test_chunk_python_file_handles_syntax_errors():
    """Test that invalid Python falls back to generic."""
    from chinvex.chunking import chunk_python_file

    text = "def broken(\n  # Missing closing paren"
    chunks = chunk_python_file(text)

    # Should not crash, should return chunks
    assert len(chunks) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_v2.py::test_extract_python_boundaries -v`
Expected: FAIL with "function not defined"

**Step 3: Implement Python AST-based chunking**

Edit `src/chinvex/chunking.py`, add:

```python
import ast


def extract_python_boundaries(text: str) -> list[int]:
    """
    Extract line numbers where top-level definitions start.

    Boundary rules:
    - Decorators stay with their function/class
    - Module docstrings are separate
    - Only top-level definitions (nested excluded)
    - if __name__ == "__main__": IS a boundary

    Returns:
        List of 0-indexed line numbers where boundaries occur
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    boundaries = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Use decorator line if present, otherwise def/class line
            start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            boundaries.append(start_line - 1)  # Convert to 0-indexed
        elif isinstance(node, ast.If):
            # Check for if __name__ == "__main__":
            if (isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == '__name__'):
                boundaries.append(node.lineno - 1)

    return sorted(boundaries)


def line_to_char_pos(text: str, line_num: int) -> int:
    """Convert 0-indexed line number to character position."""
    lines = text.split('\n')
    return sum(len(line) + 1 for line in lines[:line_num])


def chunk_code_fallback(text: str, max_chars: int = 3000) -> list[Chunk]:
    """
    Fallback chunking for code files that can't be parsed.

    WITHOUT overlap. Prefers splitting at code-safe boundaries:
    1. Blank lines (best - likely between functions/blocks)
    2. Lines ending with } or ; (good - statement/block ends)
    3. Any line boundary (ok - preserves line integrity)
    4. Character split (last resort - only for minified/absurdly long lines)

    This keeps code chunks readable and retrievable even when parsing fails.
    """
    chunks = []
    start = 0
    ordinal = 0
    ABSURDLY_LONG_LINE = 500  # If single line exceeds this, allow char split

    def find_code_split(text: str, start: int, target: int) -> int:
        """Find best split point near target for code."""
        # Search window: ±200 chars around target
        search_start = max(start, target - 200)
        search_end = min(len(text), target + 200)

        best_pos = target
        best_priority = 0

        # Priority 1: Blank lines (likely between functions)
        for i in range(search_start, search_end):
            if text[i] == '\n':
                # Check if it's a blank line (previous and next chars are \n)
                if i > 0 and i < len(text) - 1:
                    if text[i-1] == '\n' or (i+1 < len(text) and text[i+1] == '\n'):
                        distance = abs(i - target)
                        priority = 100 - (distance / 200 * 50)
                        if priority > best_priority:
                            best_priority = priority
                            best_pos = i + 1  # Start next chunk after newline

        # Priority 2: Lines ending with } or ;
        if best_priority < 50:
            for i in range(search_start, search_end):
                if i < len(text) - 1 and text[i+1] == '\n':
                    if text[i] in ('}', ';'):
                        distance = abs(i - target)
                        priority = 75 - (distance / 200 * 50)
                        if priority > best_priority:
                            best_priority = priority
                            best_pos = i + 2  # After closing char and newline

        # Priority 3: Any line boundary
        if best_priority < 25:
            for i in range(search_start, search_end):
                if text[i] == '\n':
                    distance = abs(i - target)
                    priority = 50 - (distance / 200 * 50)
                    if priority > best_priority:
                        best_priority = priority
                        best_pos = i + 1

        # Priority 4: Character split (only if no line breaks found)
        if best_priority == 0:
            best_pos = target

        return best_pos

    while start < len(text):
        target_end = start + max_chars

        if target_end >= len(text):
            # Last chunk
            chunks.append(Chunk(
                text=text[start:],
                ordinal=ordinal,
                char_start=start,
                char_end=len(text),
            ))
            break

        # Check if we're in an absurdly long line (minified code)
        next_newline = text.find('\n', start)
        if next_newline == -1 or (next_newline - start) > ABSURDLY_LONG_LINE:
            # Minified or no line breaks - just split at target
            split_pos = target_end
        else:
            # Normal code - find best split point
            split_pos = find_code_split(text, start, target_end)

        chunks.append(Chunk(
            text=text[start:split_pos],
            ordinal=ordinal,
            char_start=start,
            char_end=split_pos,
        ))

        ordinal += 1
        start = split_pos  # NO overlap - move to boundary

    return chunks if chunks else [Chunk(text=text, ordinal=0, char_start=0, char_end=len(text))]


def chunk_python_file(text: str, max_chars: int = 3000, max_lines: int = 300) -> list[Chunk]:
    """
    Split Python file respecting function/class boundaries.

    Algorithm:
    1. Extract boundary positions (where functions/classes start)
    2. Walk through boundaries, accumulating segments
    3. When accumulated size exceeds max_chars OR max_lines, flush
    4. Each chunk starts at a boundary position

    IMPORTANT: NO overlap between chunks. Semantic boundaries define chunk splits.

    Falls back to code_fallback (no overlap) if:
    - No boundaries found (syntax error or single large block)
    - AST parsing fails
    """
    boundaries = extract_python_boundaries(text)

    if not boundaries:
        # No boundaries or syntax error - fall back to code fallback (NO overlap)
        return chunk_code_fallback(text, max_chars=max_chars)

    # Convert line numbers to char positions
    boundary_positions = [0] + [line_to_char_pos(text, ln) for ln in boundaries] + [len(text)]

    if len(boundary_positions) <= 2:
        # No internal boundaries, fallback to code fallback (NO overlap)
        return chunk_code_fallback(text, max_chars=max_chars)

    chunks = []
    chunk_start = 0
    ordinal = 0

    def exceeds_limits(start: int, end: int) -> bool:
        """Check if segment exceeds either char or line limit."""
        segment = text[start:end]
        return len(segment) > max_chars or segment.count('\n') > max_lines

    for i in range(1, len(boundary_positions)):
        pos = boundary_positions[i]

        if exceeds_limits(chunk_start, pos) and chunk_start != boundary_positions[i-1]:
            # Flush chunk ending at previous boundary
            chunk_end = boundary_positions[i-1]
            chunks.append(Chunk(
                text=text[chunk_start:chunk_end],
                ordinal=ordinal,
                char_start=chunk_start,
                char_end=chunk_end,
            ))
            ordinal += 1
            chunk_start = chunk_end

    # Final chunk
    if chunk_start < len(text):
        chunks.append(Chunk(
            text=text[chunk_start:],
            ordinal=ordinal,
            char_start=chunk_start,
            char_end=len(text),
        ))

    return chunks if chunks else [Chunk(text=text, ordinal=0, char_start=0, char_end=len(text))]
```

**Step 4: Run tests**

Run: `pytest tests/test_chunking_v2.py -k "python" -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_chunking_v2.py
git commit -m "feat(chunking): add Python AST-based code-aware splitting

- Extract boundaries at top-level functions/classes
- Handle decorators (stay with function)
- Detect if __name__ == '__main__' blocks
- Fallback to generic on syntax errors

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: JS/TS heuristic splitting

**Files:**
- Modify: `tests/test_chunking_v2.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Write failing test**

Add to `tests/test_chunking_v2.py`:

```python
def test_chunk_js_file_heuristic():
    """Test JS/TS chunking using regex heuristics."""
    from chinvex.chunking import chunk_js_file

    text = '''
export function functionOne() {
  // Implementation
  return 1;
}

const functionTwo = async () => {
  // Implementation
  return 2;
};

export default class MyClass {
  method() {
    // Implementation
  }
}

module.exports = { functionOne, functionTwo };
'''

    chunks = chunk_js_file(text, max_chars=500)  # Small chunks to force splits

    # Should create multiple chunks at function/export boundaries
    assert len(chunks) >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_v2.py::test_chunk_js_file_heuristic -v`
Expected: FAIL with "function not defined"

**Step 3: Implement JS/TS heuristic chunking**

Edit `src/chinvex/chunking.py`, add:

```python
def chunk_js_file(text: str, max_chars: int = 3000) -> list[Chunk]:
    """
    Chunk JavaScript/TypeScript file using heuristic boundaries.

    Uses regex patterns to find likely function/class/export boundaries.
    NOT a full parser - may split incorrectly on edge cases.

    IMPORTANT: NO overlap between chunks. Semantic boundaries define chunk splits.
    """
    chunks = []
    start = 0
    ordinal = 0

    while start < len(text):
        target_end = start + max_chars

        if target_end >= len(text):
            # Last chunk
            chunks.append(Chunk(
                text=text[start:],
                ordinal=ordinal,
                char_start=start,
                char_end=len(text),
            ))
            break

        # Find best split using semantic boundaries (no overlap applied)
        split_pos = find_best_split(text, target_end, max_chars)

        chunks.append(Chunk(
            text=text[start:split_pos],
            ordinal=ordinal,
            char_start=start,
            char_end=split_pos,
        ))

        ordinal += 1
        start = split_pos  # NO overlap - next chunk starts at boundary

    return chunks if chunks else [Chunk(text=text, ordinal=0, char_start=0, char_end=len(text))]
```

**Step 4: Run tests**

Run: `pytest tests/test_chunking_v2.py::test_chunk_js_file_heuristic -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_chunking_v2.py
git commit -m "feat(chunking): add JS/TS heuristic code splitting

- Use regex patterns for function/class/export boundaries
- Reuse semantic boundary detection with JS patterns
- No full parser (tree-sitter deferred to future)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 8: Language detection and dispatcher

**Files:**
- Modify: `tests/test_chunking_v2.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Write failing test**

Add to `tests/test_chunking_v2.py`:

```python
def test_detect_language():
    """Test file extension to language mapping."""
    from chinvex.chunking import detect_language

    assert detect_language("file.py") == "python"
    assert detect_language("file.js") == "javascript"
    assert detect_language("file.ts") == "typescript"
    assert detect_language("file.md") == "markdown"
    assert detect_language("file.txt") == "generic"
    assert detect_language("file.unknown") == "generic"


def test_chunk_file_dispatcher():
    """Test that chunk_file dispatches to correct chunker."""
    from chinvex.chunking import chunk_file

    # Python file
    py_text = "def func():\n    pass\n" * 100
    py_chunks = chunk_file(py_text, "test.py")
    assert len(py_chunks) >= 1

    # Markdown file
    md_text = "# Header\n\n" + ("Content\n" * 500)
    md_chunks = chunk_file(md_text, "test.md")
    assert len(md_chunks) >= 1

    # Generic file
    txt_text = "Generic text " * 500
    txt_chunks = chunk_file(txt_text, "test.txt")
    assert len(txt_chunks) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_v2.py::test_detect_language -v`
Expected: FAIL with "function not defined"

**Step 3: Implement language detection and dispatcher**

Edit `src/chinvex/chunking.py`, add:

```python
from pathlib import Path


LANGUAGE_MAP = {
    '.py': 'python',
    '.pyw': 'python',
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.jsx': 'javascript',
    '.md': 'markdown',
    '.mdx': 'markdown',
}


def detect_language(filepath: str) -> str:
    """Detect language from file extension."""
    ext = Path(filepath).suffix.lower()
    return LANGUAGE_MAP.get(ext, 'generic')


def chunk_file(text: str, filepath: str, max_chars: int = 3000, max_lines: int = 300) -> list[Chunk]:
    """
    Chunk file using language-appropriate strategy.

    Dispatcher that routes to:
    - chunk_python_file() for .py - NO overlap, uses AST boundaries
    - chunk_js_file() for .js/.ts - NO overlap, uses regex boundaries
    - chunk_markdown_file() for .md - NO overlap, uses header boundaries
    - chunk_generic_file() for .txt/unknown - WITH overlap (prose/generic only)

    IMPORTANT: Overlap is ONLY applied to generic/prose files.
    Code and markdown files use semantic boundaries without overlap.
    """
    language = detect_language(filepath)

    if language == 'python':
        # Code-aware: NO overlap, uses AST boundaries
        return chunk_python_file(text, max_chars=max_chars, max_lines=max_lines)
    elif language in ('javascript', 'typescript'):
        # Code-aware: NO overlap, uses regex heuristic boundaries
        return chunk_js_file(text, max_chars=max_chars)
    elif language == 'markdown':
        # Markdown-aware: NO overlap, uses header/section boundaries
        return chunk_markdown_file(text, max_chars=max_chars)
    else:
        # Generic prose/text: ONLY case where overlap is applied
        return chunk_generic_file(text, size=max_chars, overlap=300)
```

**Step 4: Run tests**

Run: `pytest tests/test_chunking_v2.py -k "language or dispatcher" -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_chunking_v2.py
git commit -m "feat(chunking): add language detection and dispatcher

- Map file extensions to language types
- Dispatch to appropriate chunker (Python/JS/MD/generic)
- Configure max_chars and max_lines per chunker

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Update ingest to use chunker v2

**Files:**
- Modify: `src/chinvex/ingest.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Bump chunker version constant**

Edit `src/chinvex/chunking.py`, add near top:

```python
CHUNKER_VERSION = 2  # Was 1 in P0-P2
```

**Step 2: Add chunk_key function (for embedding reuse)**

Edit `src/chinvex/chunking.py`, add:

```python
import hashlib


def chunk_key(text: str) -> str:
    """
    Generate stable key for chunk embedding lookup.

    Normalizes whitespace before hashing to handle minor formatting differences.

    Returns:
        16-character hex string (sha256 prefix)
    """
    # Collapse all whitespace to single spaces
    normalized = ' '.join(text.split())
    # Hash normalized text
    hash_bytes = hashlib.sha256(normalized.encode('utf-8')).digest()
    # Return first 16 hex chars (64 bits)
    return hash_bytes.hex()[:16]
```

**Step 3: Update storage schema for chunk_key**

Edit `src/chinvex/storage.py`, add migration:

```python
def migrate_schema_v2_to_v3(conn: sqlite3.Connection):
    """
    Migrate schema from v2 to v3.

    Adds chunk_key column to chunks table for embedding reuse.
    """
    cursor = conn.cursor()

    # Check if migration already applied
    cursor.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in cursor.fetchall()]
    if "chunk_key" in columns:
        return  # Already migrated

    # Add chunk_key column
    cursor.execute("ALTER TABLE chunks ADD COLUMN chunk_key TEXT")

    # Create index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_key ON chunks(chunk_key)")

    conn.commit()
    print("Migrated schema to v3: added chunk_key column")
```

**Step 4: Update repo ingest with embedding reuse**

Edit `src/chinvex/ingest.py`:

```python
from chinvex.chunking import chunk_file, chunk_key, CHUNKER_VERSION

def ingest_repo_file(ctx, filepath: str, ...):
    """
    Ingest a single repo file.

    IMPORTANT: When chunker_version changes, automatically reuses embeddings
    for chunks with matching chunk_key to avoid re-embedding the universe.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Use new chunker v2
    new_chunks = chunk_file(text, filepath, config=ctx.config.chunking)

    if not new_chunks:
        # File skipped (too large)
        return

    # Check if this is a rechunk scenario (version changed but text similar)
    old_fingerprint = get_fingerprint(ctx, filepath)
    is_rechunk = (
        old_fingerprint and
        old_fingerprint.get('chunker_version', 1) < CHUNKER_VERSION
    )

    stats = {'embeddings_reused': 0, 'embeddings_new': 0}

    for chunk in new_chunks:
        key = chunk_key(chunk.text)

        # Try to reuse embedding if rechunking
        if is_rechunk:
            existing = lookup_chunk_by_key(ctx, key)
            if existing and hash_text(chunk.text) == existing['text_hash']:
                # Reuse existing embedding
                store_chunk_with_existing_embedding(ctx, chunk, existing['embedding'], key)
                stats['embeddings_reused'] += 1
                continue

        # Generate new embedding
        embedding = generate_embedding(chunk.text, ctx.ollama_host)
        store_chunk(ctx, chunk, embedding, key)
        stats['embeddings_new'] += 1

    # Update fingerprint with new version
    store_fingerprint(ctx, filepath, {
        'hash': hash_file(filepath),
        'chunker_version': CHUNKER_VERSION,
        'ingested_at': datetime.now().isoformat(),
    })

    # Log stats if rechunking
    if is_rechunk and (stats['embeddings_reused'] > 0 or stats['embeddings_new'] > 0):
        print(f"  Rechunked {filepath}: reused {stats['embeddings_reused']}, new {stats['embeddings_new']}")

    return stats
```

**Step 5: Add storage helpers**

Edit `src/chinvex/storage.py`:

```python
def lookup_chunk_by_key(ctx: Context, chunk_key: str) -> dict | None:
    """Lookup chunk by chunk_key for embedding reuse."""
    conn = sqlite3.connect(ctx.db_path)
    cursor = conn.execute(
        "SELECT chunk_id, text, embedding FROM chunks WHERE chunk_key = ? LIMIT 1",
        (chunk_key,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'chunk_id': row[0],
        'text': row[1],
        'text_hash': hash_text(row[1]),
        'embedding': deserialize_embedding(row[2]),
    }


def hash_text(text: str) -> str:
    """Hash text for comparison (normalized)."""
    import hashlib
    normalized = ' '.join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def store_chunk_with_existing_embedding(ctx, chunk, embedding, chunk_key):
    """Store chunk reusing an existing embedding."""
    # Same as store_chunk, but logs reuse
    store_chunk(ctx, chunk, embedding, chunk_key)
```

**Step 6: Test ingest with automatic embedding reuse**

Run: `chinvex ingest --context TestContext --full`
Expected:
- Files are chunked using new chunker
- Embeddings reused where text matches (via chunk_key)
- Log shows "Rechunked: reused X, new Y" for affected files
- No need for separate --rechunk-only flag (automatic)

**Step 7: Update fingerprint schema**

Edit `src/chinvex/storage.py`:

```python
# In fingerprint dict:
fingerprint = {
    'source_uri': source_uri,
    'hash': file_hash,
    'chunker_version': CHUNKER_VERSION,  # Track version for auto-rechunk
    'ingested_at': datetime.now().isoformat(),
}
```

**Step 8: Test version bump behavior**

Simulate version bump:
1. Ingest with CHUNKER_VERSION = 1
2. Bump to CHUNKER_VERSION = 2
3. Ingest again
4. Verify: chunks are regenerated, but embeddings are reused where possible
5. Expected output: "Rechunked: reused X, new Y" for files with matching chunks

**Step 9: Commit**

```bash
git add src/chinvex/ingest.py src/chinvex/chunking.py src/chinvex/storage.py
git commit -m "feat(ingest): integrate chunker v2 with automatic embedding reuse

- Bump CHUNKER_VERSION to 2
- Add chunk_key() for stable embedding lookup
- Add chunk_key column to chunks table (v2→v3 migration)
- Automatically reuse embeddings on version bump (no --rechunk-only needed)
- Use chunk_file() dispatcher in ingest
- Store chunker_version in fingerprints
- Log rechunk stats (reused vs new embeddings)

BREAKING: Version bump now triggers re-chunk, but reuses embeddings.
This prevents 're-embed the universe' on chunking improvements.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Add configuration for chunking

**Files:**
- Modify: `src/chinvex/config.py`
- Modify: `src/chinvex/chunking.py`

**Step 1: Add chunking config schema**

Edit `src/chinvex/config.py`:

```python
@dataclass
class ChunkingConfig:
    """
    Chunking v2 configuration.

    IMPORTANT: overlap_chars only applies to generic/prose files (.txt, unknown extensions).
    Code (Python/JS/TS) and markdown files use semantic boundaries without overlap.
    """
    max_chars: int = 3000
    max_lines: int = 300
    overlap_chars: int = 300  # Only for generic files - NOT applied to code/markdown
    semantic_boundaries: bool = True
    code_aware: bool = True
    skip_files_larger_than_mb: int = 5


@dataclass
class ContextConfig:
    """Context configuration."""
    # ... existing fields ...
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
```

**Step 2: Apply config in chunking functions**

Edit `src/chinvex/chunking.py`, modify chunk_file signature:

```python
def chunk_file(
    text: str,
    filepath: str,
    config: ChunkingConfig | None = None
) -> list[Chunk]:
    """Chunk file using language-appropriate strategy."""
    if config is None:
        config = ChunkingConfig()  # Defaults

    # Check file size limit
    file_size_mb = len(text.encode('utf-8')) / (1024 * 1024)
    if file_size_mb > config.skip_files_larger_than_mb:
        print(f"Warning: Skipping large file: {filepath} ({file_size_mb:.1f}MB > {config.skip_files_larger_than_mb}MB)")
        return []

    language = detect_language(filepath)

    if not config.code_aware:
        # Force generic chunking if code_aware disabled
        return chunk_generic_file(text, size=config.max_chars, overlap=config.overlap_chars)

    if language == 'python':
        return chunk_python_file(text, max_chars=config.max_chars, max_lines=config.max_lines)
    elif language in ('javascript', 'typescript'):
        return chunk_js_file(text, max_chars=config.max_chars)
    elif language == 'markdown':
        return chunk_markdown_file(text, max_chars=config.max_chars)
    else:
        return chunk_generic_file(text, size=config.max_chars, overlap=config.overlap_chars)
```

**Step 3: Pass config from ingest**

Edit `src/chinvex/ingest.py`:

```python
def ingest_repo_file(ctx, filepath: str, ...):
    """Ingest a single repo file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    chunks = chunk_file(text, filepath, config=ctx.config.chunking)

    if not chunks:
        # File was skipped (too large)
        return

    # ... rest of ingestion ...
```

**Step 4: Test with custom config**

Create test context config with `chunking.skip_files_larger_than_mb: 1`, verify large files skipped.

**Step 5: Commit**

```bash
git add src/chinvex/config.py src/chinvex/chunking.py src/chinvex/ingest.py
git commit -m "feat(chunking): add configuration support

- Add ChunkingConfig with tunable limits
- Support skip_files_larger_than_mb limit
- Allow disabling code_aware for fallback
- Pass config from context to chunker

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2b: Rechunk Optimization (P3.1b)

**Priority:** Optional - Core embedding reuse already integrated in Task 9

**NOTE:** Task 9 already implements automatic embedding reuse on version bump via chunk_key.
Phase 2b adds:
- Additional tests for chunk_key stability (Task 11)
- Explicit --rechunk-only CLI flag for manual control (Task 13)
- These are nice-to-haves, not blockers for P3a release

**Decision point:** You can skip Phase 2b entirely if Task 9's automatic reuse is sufficient.
Only implement if you need:
- More comprehensive tests of reuse behavior
- Explicit --rechunk-only flag for manual rechunking

---

### Task 11: Stable chunk keys for embedding reuse

**Files:**
- Create: `tests/test_rechunk_optimization.py`
- Modify: `src/chinvex/chunking.py`
- Modify: `src/chinvex/storage.py`

**Step 1: Write failing test**

Create `tests/test_rechunk_optimization.py`:

```python
"""Test rechunk optimization with embedding reuse."""
import pytest
from chinvex.chunking import chunk_key


def test_chunk_key_stable():
    """Test that chunk_key is stable for same text."""
    text1 = "This is test content."
    text2 = "This is test content."  # Same content

    assert chunk_key(text1) == chunk_key(text2)


def test_chunk_key_normalizes_whitespace():
    """Test that chunk_key normalizes whitespace."""
    text1 = "This  is   test content."
    text2 = "This is test content."

    # Should produce same key (whitespace normalized)
    assert chunk_key(text1) == chunk_key(text2)


def test_chunk_key_different_content():
    """Test that different content produces different keys."""
    text1 = "This is test content."
    text2 = "This is different content."

    assert chunk_key(text1) != chunk_key(text2)


def test_chunk_key_length():
    """Test that chunk_key produces 16-char hex string."""
    text = "Test content"
    key = chunk_key(text)

    assert len(key) == 16
    assert all(c in '0123456789abcdef' for c in key)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rechunk_optimization.py::test_chunk_key_stable -v`
Expected: FAIL with "function not defined"

**Step 3: Implement chunk_key function**

Edit `src/chinvex/chunking.py`, add:

```python
import hashlib


def chunk_key(text: str) -> str:
    """
    Generate stable key for chunk embedding lookup.

    Normalizes whitespace before hashing to handle minor formatting differences.

    Returns:
        16-character hex string (sha256 prefix)
    """
    # Collapse all whitespace to single spaces
    normalized = ' '.join(text.split())
    # Hash normalized text
    hash_bytes = hashlib.sha256(normalized.encode('utf-8')).digest()
    # Return first 16 hex chars (64 bits)
    return hash_bytes.hex()[:16]
```

**Step 4: Run tests**

Run: `pytest tests/test_rechunk_optimization.py -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_rechunk_optimization.py
git commit -m "feat(chunking): add stable chunk key for embedding reuse

- Implement chunk_key() with whitespace normalization
- Use sha256 hash prefix (16 chars)
- Foundation for rechunk optimization

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 12: Schema migration for chunk_key storage

**Files:**
- Modify: `src/chinvex/storage.py`
- Create: `tests/test_chunk_key_storage.py`

**Step 1: Write migration test**

Create `tests/test_chunk_key_storage.py`:

```python
"""Test chunk_key storage and migration."""
import pytest
import sqlite3
from pathlib import Path


def test_chunks_table_has_chunk_key_column(tmp_path):
    """Test that chunks table includes chunk_key column."""
    from chinvex.storage import init_storage

    db_path = tmp_path / "test.db"
    init_storage(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in cursor.fetchall()]

    assert "chunk_key" in columns


def test_chunk_key_index_exists(tmp_path):
    """Test that chunk_key index is created."""
    from chinvex.storage import init_storage

    db_path = tmp_path / "test.db"
    init_storage(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_chunk_key'")

    assert cursor.fetchone() is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunk_key_storage.py -v`
Expected: FAIL with "no such column: chunk_key"

**Step 3: Add migration for chunk_key column**

Edit `src/chinvex/storage.py`, find schema migration section and add:

```python
def migrate_schema_v2_to_v3(conn: sqlite3.Connection):
    """
    Migrate schema from v2 to v3.

    Adds chunk_key column to chunks table for rechunk optimization.
    """
    cursor = conn.cursor()

    # Check if migration already applied
    cursor.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in cursor.fetchall()]
    if "chunk_key" in columns:
        return  # Already migrated

    # Add chunk_key column
    cursor.execute("ALTER TABLE chunks ADD COLUMN chunk_key TEXT")

    # Create index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_key ON chunks(chunk_key)")

    conn.commit()
    print("Migrated schema to v3: added chunk_key column")


def init_storage(db_path: str):
    """Initialize storage with schema migrations."""
    conn = sqlite3.connect(db_path)

    # ... existing schema setup ...

    # Run migrations
    migrate_schema_v2_to_v3(conn)

    conn.close()
```

**Step 4: Run tests**

Run: `pytest tests/test_chunk_key_storage.py -v`
Expected: Tests pass

**Step 5: Test migration on existing database**

Run: `python -c "from chinvex.storage import init_storage; init_storage('test.db')"`
Expected: Migration runs, column added

**Step 6: Commit**

```bash
git add src/chinvex/storage.py tests/test_chunk_key_storage.py
git commit -m "feat(storage): add chunk_key column for embedding reuse

- Add ALTER TABLE migration for chunk_key
- Create index on chunk_key for fast lookups
- Idempotent migration (checks before adding)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 13: Implement --rechunk-only CLI flag

**Files:**
- Modify: `src/chinvex/cli.py`
- Modify: `src/chinvex/ingest.py`
- Create: `tests/test_rechunk_only.py`

**Step 1: Write failing test**

Create `tests/test_rechunk_only.py`:

```python
"""Test --rechunk-only flag."""
import pytest
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_rechunk_only_flag_accepted():
    """Test that --rechunk-only flag is recognized."""
    result = runner.invoke(app, ["ingest", "--context", "Test", "--rechunk-only"])

    # Should not fail with "no such option"
    assert "no such option" not in result.stdout.lower()


def test_rechunk_only_reuses_embeddings(tmp_path):
    """Test that --rechunk-only reuses embeddings when text unchanged."""
    # This is an integration test - implementation will verify behavior
    # For now, just test that flag is processed
    pass
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rechunk_only.py::test_rechunk_only_flag_accepted -v`
Expected: FAIL with "no such option"

**Step 3: Add --rechunk-only flag to ingest command**

Edit `src/chinvex/cli.py`, find `ingest` command:

```python
@app.command()
def ingest(
    context: str = Option(..., "--context", "-c", help="Context name"),
    full: bool = Option(False, "--full", help="Full reingest (ignore fingerprints)"),
    rechunk_only: bool = Option(False, "--rechunk-only", help="Rechunk only, reuse embeddings when possible"),
    ollama_host: str = Option("http://localhost:11434", "--ollama-host", help="Ollama API host"),
):
    """Ingest content into context."""
    from chinvex.ingest import run_ingest
    from chinvex.context import load_context

    ctx = load_context(context)

    result = run_ingest(
        ctx=ctx,
        full=full,
        rechunk_only=rechunk_only,
        ollama_host=ollama_host,
    )

    print(f"Ingest complete: {result['chunks_added']} chunks added")
    if rechunk_only and 'embeddings_reused' in result:
        print(f"Rechunk optimization: {result['embeddings_reused']} embeddings reused, {result['embeddings_new']} new")
```

**Step 4: Implement rechunk_only logic in ingest**

Edit `src/chinvex/ingest.py`:

```python
def run_ingest(
    ctx: Context,
    full: bool = False,
    rechunk_only: bool = False,
    ollama_host: str = "http://localhost:11434",
) -> dict:
    """
    Run ingestion with optional rechunk optimization.

    When rechunk_only=True:
    - Run new chunker on all files
    - For each chunk, compute chunk_key
    - Check if chunk_key exists in DB with same text hash
    - If yes: reuse existing embedding (don't re-embed)
    - If no: generate new embedding
    """
    stats = {
        'chunks_added': 0,
        'embeddings_reused': 0,
        'embeddings_new': 0,
    }

    # ... existing ingest logic ...

    if rechunk_only:
        # Rechunk optimization path
        for source_file in source_files:
            text = read_file(source_file)
            new_chunks = chunk_file(text, source_file, config=ctx.config.chunking)

            for chunk in new_chunks:
                key = chunk_key(chunk.text)

                # Check if chunk_key exists with same text
                existing = lookup_chunk_by_key(ctx, key)

                if existing and existing['text_hash'] == hash_text(chunk.text):
                    # Reuse existing embedding
                    reuse_embedding(ctx, chunk, existing['embedding'])
                    stats['embeddings_reused'] += 1
                else:
                    # Generate new embedding
                    embedding = generate_embedding(chunk.text, ollama_host)
                    store_chunk(ctx, chunk, embedding, key)
                    stats['embeddings_new'] += 1

                stats['chunks_added'] += 1

    return stats
```

**Step 5: Implement helper functions**

Edit `src/chinvex/storage.py`, add:

```python
def lookup_chunk_by_key(ctx: Context, chunk_key: str) -> dict | None:
    """Lookup chunk by chunk_key for embedding reuse."""
    conn = sqlite3.connect(ctx.db_path)
    cursor = conn.execute(
        "SELECT chunk_id, text, text_hash FROM chunks WHERE chunk_key = ?",
        (chunk_key,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'chunk_id': row[0],
        'text': row[1],
        'text_hash': row[2],
    }


def hash_text(text: str) -> str:
    """Hash text for comparison (normalized)."""
    import hashlib
    normalized = ' '.join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()
```

**Step 6: Run tests**

Run: `pytest tests/test_rechunk_only.py -v`
Expected: Tests pass

**Step 7: Test manually**

Run: `chinvex ingest --context Test --rechunk-only`
Expected: Stats show embeddings reused vs new

**Step 8: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/ingest.py src/chinvex/storage.py tests/test_rechunk_only.py
git commit -m "feat(ingest): add --rechunk-only flag with embedding reuse

- Add --rechunk-only CLI flag
- Lookup chunks by chunk_key for reuse
- Log stats: embeddings_reused vs embeddings_new
- Avoid re-embedding unchanged text

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## P3b: Proactive Foundation

### Phase 3: Watch History (P3.2 — log only)

---

### Task 14: Watch history log appending

**Files:**
- Create: `tests/watch/test_watch_history.py`
- Modify: `src/chinvex/watch/runner.py`

**Step 1: Write failing test**

Create `tests/watch/test_watch_history.py`:

```python
"""Test watch history logging."""
import pytest
import json
from pathlib import Path


def test_append_watch_history(tmp_path):
    """Test that watch hits are appended to history log."""
    from chinvex.watch.runner import append_watch_history

    history_file = tmp_path / "watch_history.jsonl"

    entry = {
        "ts": "2026-01-28T12:00:00Z",
        "run_id": "run_123",
        "watch_id": "test_watch",
        "query": "test query",
        "hits": [
            {"chunk_id": "abc", "score": 0.85, "snippet": "test snippet"}
        ]
    }

    append_watch_history(str(history_file), entry)

    # Verify file exists and contains entry
    assert history_file.exists()
    with open(history_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["watch_id"] == "test_watch"


def test_append_watch_history_multiple_entries(tmp_path):
    """Test that multiple watch hits are appended."""
    from chinvex.watch.runner import append_watch_history

    history_file = tmp_path / "watch_history.jsonl"

    for i in range(3):
        entry = {
            "ts": f"2026-01-28T12:0{i}:00Z",
            "run_id": f"run_{i}",
            "watch_id": "test_watch",
            "query": "test",
            "hits": []
        }
        append_watch_history(str(history_file), entry)

    with open(history_file) as f:
        lines = f.readlines()
    assert len(lines) == 3


def test_watch_history_caps_hits_at_10(tmp_path):
    """Test that watch history caps hits at 10 per entry."""
    from chinvex.watch.runner import create_watch_history_entry

    # Create entry with 20 hits
    hits = [{"chunk_id": f"chunk_{i}", "score": 0.9 - i*0.01, "snippet": f"text {i}"}
            for i in range(20)]

    entry = create_watch_history_entry(
        watch_id="test",
        query="test",
        hits=hits,
        run_id="run_123"
    )

    # Should cap at 10 and mark truncated
    assert len(entry["hits"]) == 10
    assert entry["truncated"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/watch/test_watch_history.py::test_append_watch_history -v`
Expected: FAIL with "function not defined"

**Step 3: Implement watch history logging**

Edit `src/chinvex/watch/runner.py`, add:

```python
import json
from datetime import datetime
from pathlib import Path


def append_watch_history(history_file: str, entry: dict):
    """
    Append watch history entry to JSONL log.

    Creates file if it doesn't exist.
    """
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def create_watch_history_entry(
    watch_id: str,
    query: str,
    hits: list[dict],
    run_id: str
) -> dict:
    """
    Create watch history entry with hit capping.

    Caps hits at 10 and marks as truncated if exceeded.
    """
    truncated = len(hits) > 10
    capped_hits = hits[:10]

    # Extract snippet (first 200 chars) from each hit
    formatted_hits = []
    for hit in capped_hits:
        formatted_hits.append({
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "snippet": hit.get("text", "")[:200]
        })

    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "watch_id": watch_id,
        "query": query,
        "hits": formatted_hits,
    }

    if truncated:
        entry["truncated"] = True

    return entry


def log_watch_hits(ctx: Context, watch: Watch, hits: list[SearchResult], run_id: str):
    """
    Log watch hits to history file.

    Called after watch triggers during ingest.
    """
    history_file = Path(ctx.base_dir) / "watch_history.jsonl"

    entry = create_watch_history_entry(
        watch_id=watch.id,
        query=watch.query,
        hits=[h.to_dict() for h in hits],
        run_id=run_id,
    )

    append_watch_history(str(history_file), entry)
```

**Step 4: Integrate with watch runner**

Edit `src/chinvex/watch/runner.py`, find watch execution logic:

```python
def run_watches(ctx: Context, run_id: str):
    """Run all watches and log hits."""
    for watch in ctx.config.watches:
        hits = search_hybrid(ctx, watch.query, k=20)

        if hits:
            # Log to history
            log_watch_hits(ctx, watch, hits, run_id)

            # ... existing notification logic ...
```

**Step 5: Run tests**

Run: `pytest tests/watch/test_watch_history.py -v`
Expected: Tests pass

**Step 6: Commit**

```bash
git add src/chinvex/watch/runner.py tests/watch/test_watch_history.py
git commit -m "feat(watch): add history logging to JSONL

- Append watch hits to watch_history.jsonl
- Cap hits at 10 per entry with truncation flag
- Include snippet (first 200 chars)
- Integrate with watch runner

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 15: CLI watch history commands

**Files:**
- Modify: `src/chinvex/cli.py`
- Create: `tests/test_watch_history_cli.py`

**Step 1: Write failing test**

Create `tests/test_watch_history_cli.py`:

```python
"""Test watch history CLI commands."""
import pytest
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_watch_history_command():
    """Test chinvex watch history command."""
    result = runner.invoke(app, ["watch", "history", "--context", "Test"])
    assert result.exit_code == 0


def test_watch_history_with_filters():
    """Test watch history with filters."""
    result = runner.invoke(app, [
        "watch", "history",
        "--context", "Test",
        "--since", "7d",
        "--id", "test_watch",
        "--limit", "10"
    ])
    assert result.exit_code == 0


def test_watch_history_formats():
    """Test watch history output formats."""
    # JSON format
    result = runner.invoke(app, [
        "watch", "history",
        "--context", "Test",
        "--format", "json"
    ])
    assert result.exit_code == 0

    # Table format (default)
    result = runner.invoke(app, [
        "watch", "history",
        "--context", "Test",
        "--format", "table"
    ])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_watch_history_cli.py::test_watch_history_command -v`
Expected: FAIL with "no such command"

**Step 3: Add watch history command**

Edit `src/chinvex/cli.py`, add:

```python
@app.command()
def watch(
    action: str = Argument(..., help="Action: history, clear"),
    context: str = Option(..., "--context", "-c", help="Context name"),
    since: str | None = Option(None, "--since", help="Filter by time (e.g., 7d, 1h)"),
    id: str | None = Option(None, "--id", help="Filter by watch ID"),
    limit: int = Option(50, "--limit", help="Maximum entries to show"),
    format: str = Option("table", "--format", help="Output format: table, json"),
):
    """Manage watch history."""
    from chinvex.context import load_context
    from chinvex.watch.history import read_watch_history, format_history_table, format_history_json
    from datetime import datetime, timedelta

    ctx = load_context(context)

    if action == "history":
        # Parse since filter
        since_ts = None
        if since:
            since_ts = parse_time_delta(since)

        # Read history
        entries = read_watch_history(
            ctx,
            since=since_ts,
            watch_id=id,
            limit=limit,
        )

        # Format output
        if format == "json":
            print(format_history_json(entries))
        else:
            print(format_history_table(entries))

    elif action == "clear":
        # TODO: Implement clear
        print("Clear not yet implemented")

    else:
        print(f"Unknown action: {action}")
        raise typer.Exit(1)


def parse_time_delta(s: str) -> datetime:
    """Parse time delta string like '7d', '1h' into datetime."""
    from datetime import datetime, timedelta

    if s.endswith('d'):
        days = int(s[:-1])
        return datetime.utcnow() - timedelta(days=days)
    elif s.endswith('h'):
        hours = int(s[:-1])
        return datetime.utcnow() - timedelta(hours=hours)
    elif s.endswith('m'):
        minutes = int(s[:-1])
        return datetime.utcnow() - timedelta(minutes=minutes)
    else:
        raise ValueError(f"Invalid time delta: {s}")
```

**Step 4: Implement history reading and formatting**

Create `src/chinvex/watch/history.py`:

```python
"""Watch history reading and formatting."""
import json
from datetime import datetime
from pathlib import Path


def read_watch_history(
    ctx,
    since: datetime | None = None,
    watch_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Read watch history with filters.

    Returns list of history entries (most recent first).
    """
    history_file = Path(ctx.base_dir) / "watch_history.jsonl"

    if not history_file.exists():
        return []

    entries = []
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            # Apply filters
            if since:
                entry_ts = datetime.fromisoformat(entry["ts"].rstrip("Z"))
                if entry_ts < since:
                    continue

            if watch_id and entry["watch_id"] != watch_id:
                continue

            entries.append(entry)

    # Most recent first
    entries.reverse()

    # Apply limit
    return entries[:limit]


def format_history_table(entries: list[dict]) -> str:
    """Format history as ASCII table."""
    if not entries:
        return "No watch history found."

    lines = []
    lines.append(f"{'Timestamp':<20} {'Watch ID':<15} {'Hits':<5} {'Query':<30}")
    lines.append("-" * 75)

    for entry in entries:
        ts = entry["ts"][:19]  # Strip milliseconds
        watch_id = entry["watch_id"][:14]
        hits = len(entry["hits"])
        query = entry["query"][:29]

        lines.append(f"{ts:<20} {watch_id:<15} {hits:<5} {query:<30}")

    return "\n".join(lines)


def format_history_json(entries: list[dict]) -> str:
    """Format history as JSON."""
    return json.dumps(entries, indent=2)
```

**Step 5: Run tests**

Run: `pytest tests/test_watch_history_cli.py -v`
Expected: Tests pass

**Step 6: Test manually**

Run: `chinvex watch history --context Test --format table`
Expected: Table of watch history displayed

**Step 7: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/watch/history.py tests/test_watch_history_cli.py
git commit -m "feat(cli): add watch history commands

- Add 'chinvex watch history' command
- Support --since, --id, --limit filters
- Support --format table/json output
- Implement history reading and formatting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## P3c: Policy + Ops

### Phase 4: Archive Tier (P3.4)

---

### Task 16: Archive schema migration

**Files:**
- Modify: `src/chinvex/storage.py`
- Create: `tests/test_archive_schema.py`

**Step 1: Write failing test**

Create `tests/test_archive_schema.py`:

```python
"""Test archive schema migration."""
import pytest
import sqlite3


def test_documents_table_has_archived_column(tmp_path):
    """Test that documents table has archived column."""
    from chinvex.storage import init_storage

    db_path = tmp_path / "test.db"
    init_storage(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(documents)")
    columns = [row[1] for row in cursor.fetchall()]

    assert "archived" in columns
    assert "archived_at" in columns


def test_archived_index_exists(tmp_path):
    """Test that archived index is created."""
    from chinvex.storage import init_storage

    db_path = tmp_path / "test.db"
    init_storage(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_documents_archived'"
    )

    assert cursor.fetchone() is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_archive_schema.py -v`
Expected: FAIL with "no such column: archived"

**Step 3: Add archive schema migration**

Edit `src/chinvex/storage.py`:

```python
def migrate_schema_v3_to_v4(conn: sqlite3.Connection):
    """
    Migrate schema from v3 to v4.

    Adds archived and archived_at columns to documents table.
    """
    cursor = conn.cursor()

    # Check if migration already applied
    cursor.execute("PRAGMA table_info(documents)")
    columns = [row[1] for row in cursor.fetchall()]
    if "archived" in columns:
        return  # Already migrated

    # Add archived columns
    cursor.execute("ALTER TABLE documents ADD COLUMN archived INTEGER DEFAULT 0")
    cursor.execute("ALTER TABLE documents ADD COLUMN archived_at TEXT")

    # Create index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_archived ON documents(archived)")

    conn.commit()
    print("Migrated schema to v4: added archive tier support")


def init_storage(db_path: str):
    """Initialize storage with schema migrations."""
    conn = sqlite3.connect(db_path)

    # ... existing schema setup ...

    # Run migrations
    migrate_schema_v2_to_v3(conn)
    migrate_schema_v3_to_v4(conn)  # NEW

    conn.close()
```

**Step 4: Run tests**

Run: `pytest tests/test_archive_schema.py -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/storage.py tests/test_archive_schema.py
git commit -m "feat(storage): add archive tier schema migration

- Add archived and archived_at columns to documents
- Create index on archived for fast filtering
- Idempotent migration (v3 to v4)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

*(Due to length constraints, I'll provide the remaining tasks in summary form with key implementation points)*

### Task 17: Archive run command (dry-run + execute)

- Add `chinvex archive run` CLI command
- Implement `--older-than` filter based on `updated_at` or `ingested_at`
- Dry-run by default, `--force` to execute
- Update `documents.archived = 1` and `archived_at = now()`

### Task 18: Search filtering with archive penalty

- Modify `search_hybrid()` to filter out `archived = 1` by default
- Add `include_archive` parameter
- Apply `archive_penalty` multiplier when `include_archive = True`
- Ensure recency decay NOT applied to archived docs (penalty replaces it)

### Task 19: Archive list and restore commands

- Add `chinvex archive list` to show archived documents
- Add `chinvex archive restore --doc-id` to flip `archived = 0`
- Restore does NOT re-ingest or re-embed

### Task 20: Auto-archive on ingest

- Add post-ingest hook to check `age_threshold_days`
- Use `get_doc_age_timestamp()` helper (updated_at or ingested_at)
- Log: `"Archived N docs (older than X days)"`

### Task 21: Archive purge command

- Add `chinvex archive purge --older-than --force`
- Permanently delete archived docs older than threshold
- Dry-run by default

---

### Phase 5: Webhooks (P3.2b)

### Task 22: Webhook notification implementation

- Add `notifications` config section
- Implement `send_webhook()` with retry logic
- HTTPS validation and private IP blocking
- Snippet-only payload (first 200 chars), sanitize source_uri to filename

### Task 23: Webhook signature generation

- Add HMAC-SHA256 signature to payload
- Store secret in `env:CHINVEX_WEBHOOK_SECRET`
- Include signature in `X-Chinvex-Signature` header

### Task 24: Integrate webhooks with watch runner

- Call `send_webhook()` when watch triggers
- Check `min_score_for_notify` threshold
- Continue on webhook failure (don't block ingest)

---

### Phase 6: Gateway Extras (P3.5)

### Task 25: Redis-backed rate limiting

- Add optional Redis backend for rate limiting
- Fallback to in-memory if Redis unavailable
- Log warning on fallback

### Task 26: Prometheus metrics endpoint

- Add `/metrics` endpoint
- Expose request counts, latency histograms, grounded ratio
- Require same bearer token as other endpoints
- Ephemeral metrics (reset on restart)

---

## Configuration Summary

Final P3 config structure:

```json
{
  "chunking": {
    "max_chars": 3000,
    "max_lines": 300,
    "overlap_chars": 300,
    "semantic_boundaries": true,
    "code_aware": true,
    "skip_files_larger_than_mb": 5
  },
  "notifications": {
    "enabled": false,
    "webhook_url": "",
    "webhook_secret": "env:CHINVEX_WEBHOOK_SECRET",
    "notify_on": ["watch_hit"],
    "min_score_for_notify": 0.75,
    "retry_count": 2,
    "retry_delay_sec": 5
  },
  "cross_context": {
    "enabled": true,
    "max_contexts_per_query": 10,
    "k_per_context": 20
  },
  "archive": {
    "enabled": true,
    "age_threshold_days": 180,
    "auto_archive_on_ingest": true,
    "archive_penalty": 0.8
  },
  "gateway": {
    "rate_limit": {
      "backend": "memory",
      "redis_url": null,
      "requests_per_minute": 60,
      "requests_per_hour": 500
    },
    "metrics_enabled": true,
    "metrics_auth_required": true
  }
}
```

---

## Testing Strategy

### Unit Tests
- All new functions have dedicated unit tests
- Use pytest fixtures for test contexts
- Mock external dependencies (Ollama, Redis)

### Integration Tests
- End-to-end CLI command tests
- Gateway API contract tests
- Cross-context search with real indexes

### Manual Acceptance Tests
Run through acceptance tests from spec Section 13 after implementation.

---

## Deployment

### P2 → P3 Upgrade

```bash
# 1. Backup
cp -r P:/ai_memory/ P:/ai_memory_backup_$(date +%Y%m%d)/

# 2. Pull and install
git pull origin main
pip install -r requirements.txt --break-system-packages

# 3. Run migrations (auto on first command)
chinvex version

# 4. Re-ingest with new chunker (recommended)
chinvex ingest --context Chinvex --full

# 5. Restart gateway
pm2 restart chinvex-gateway
```

### Rollback Plan
```bash
git checkout v0.2.0
pip install -r requirements.txt --break-system-packages
# Database migrations are backward compatible
```

---

## Plan Complete

**Total estimated effort:** 8-10 days across 6 phases

**Critical path:**
1. Cross-context search (quick win) → 1-2 days
2. Chunking v2 (requires re-index) → 2-3 days
3. Archive tier (policy foundation) → 2 days
4. Watch history + webhooks + gateway extras → 3-4 days

---

## Execution Options

Plan complete and saved to `docs/plans/2026-01-28-p3-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints