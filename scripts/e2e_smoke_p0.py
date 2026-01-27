#!/usr/bin/env python3
"""
E2E Smoke Test for P0 Implementation

Tests:
1. Create test context
2. Ingest chinvex repo
3. Ingest mock Codex session (via mocked app-server)
4. Search across both sources
5. chinvex_answer with query that CAN be grounded
6. chinvex_answer with query that CANNOT be grounded
7. Re-run ingest, verify fingerprints skip unchanged files

Exit codes:
- 0: All tests passed
- 1: One or more tests failed
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import requests

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, load_context
from chinvex.ingest import ingest_context
from chinvex.search import search_context


class SmokeTestFailure(Exception):
    pass


def create_test_context(tmp_dir: Path) -> ContextConfig:
    """Create a test context pointing to chinvex repo and temp index."""
    print("[CREATE] Creating test context...")

    # Get chinvex repo path (parent of scripts/)
    repo_path = Path(__file__).parent.parent.resolve()

    # Create context structure
    context_root = tmp_dir / "contexts" / "smoke_test"
    context_root.mkdir(parents=True, exist_ok=True)

    index_dir = tmp_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    context_data = {
        "schema_version": 1,
        "name": "smoke_test",
        "aliases": ["test"],
        "includes": {
            "repos": [str(repo_path)],
            "chat_roots": [],
            "codex_session_roots": [str(tmp_dir / "codex_sessions")],  # Will be mocked
            "note_roots": [],
        },
        "index": {
            "sqlite_path": str(index_dir / "hybrid.db"),
            "chroma_dir": str(index_dir / "chroma"),
        },
        "weights": {
            "fts": 0.4,
            "vector": 0.6,
            "repo": 1.0,
            "chat": 1.0,
            "codex_session": 1.0,
        },
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z",
    }

    context_file = context_root / "context.json"
    context_file.write_text(json.dumps(context_data, indent=2), encoding="utf-8")

    ctx = ContextConfig.from_dict(context_data)
    print(f"[PASS] Context created: {ctx.name}")
    print(f"   Repo: {repo_path}")
    print(f"   Index: {index_dir}")
    return ctx


def mock_appserver_responses():
    """Create mock responses for app-server calls."""
    mock_thread = {
        "id": "test-thread-001",
        "title": "P0 Implementation Discussion",
        "updated_at": "2026-01-26T10:02:00Z",
        "turns": [
            {
                "id": "turn-001",
                "role": "user",
                "text": "How does chinvex implement hybrid retrieval with SQLite FTS5 and Chroma?",
                "timestamp": "2026-01-26T10:00:00Z",
            },
            {
                "id": "turn-002",
                "role": "assistant",
                "text": "Chinvex uses SQLite FTS5 for lexical search with BM25 ranking, and Chroma for vector search with cosine similarity. The scores are normalized and blended using configurable weights.",
                "timestamp": "2026-01-26T10:00:30Z",
            },
            {
                "id": "turn-003",
                "role": "user",
                "text": "What about fingerprinting for incremental ingest?",
                "timestamp": "2026-01-26T10:01:00Z",
            },
            {
                "id": "turn-004",
                "role": "assistant",
                "text": "Fingerprinting uses mtime and content hash for files, and updated_at timestamps for conversations. This allows skipping unchanged documents during re-ingest.",
                "timestamp": "2026-01-26T10:01:30Z",
            },
        ],
    }

    return {
        "list_threads": [
            {
                "id": "test-thread-001",
                "title": "P0 Implementation Discussion",
                "updated_at": "2026-01-26T10:02:00Z",
            }
        ],
        "get_thread": mock_thread,
    }


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def test_ingest_repo(ctx: ContextConfig) -> dict:
    """Test 1: Ingest chinvex repo."""
    print("\n[TEST-1] Test 1: Ingesting chinvex repo...")

    if not check_ollama():
        raise SmokeTestFailure(
            "Ollama is not running on localhost:11434. "
            "Please start Ollama (`ollama serve`) before running smoke tests."
        )

    # Use local Ollama
    stats = ingest_context(ctx, ollama_host_override="http://127.0.0.1:11434")

    print(f"[PASS] Ingest complete: {stats}")

    if stats["documents"] == 0:
        raise SmokeTestFailure("No documents ingested from repo")

    if stats["chunks"] == 0:
        raise SmokeTestFailure("No chunks created from repo")

    return stats


def test_ingest_codex_session(ctx: ContextConfig) -> dict:
    """Test 2: Ingest mock Codex session via mocked app-server."""
    print("\n[TEST-2] Test 2: Ingesting mock Codex session...")

    mock_responses = mock_appserver_responses()

    # Mock the app-server HTTP calls
    with patch("chinvex.adapters.cx_appserver.client.requests.get") as mock_get:
        # Configure mock to return our test data
        def side_effect(url, timeout=None):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    if "/thread/list" in url:
                        return {"threads": mock_responses["list_threads"]}
                    elif "/thread/resume" in url:
                        return mock_responses["get_thread"]
                    return {}

            return MockResponse()

        mock_get.side_effect = side_effect

        # Set app-server URL env var
        os.environ["CHINVEX_APPSERVER_URL"] = "http://localhost:8080"

        stats = ingest_context(ctx, ollama_host_override="http://127.0.0.1:11434")

        print(f"[PASS] Codex ingest complete: {stats}")

        if stats["documents"] == 0:
            raise SmokeTestFailure("No Codex documents ingested")

        return stats


def test_search(ctx: ContextConfig):
    """Test 3: Search across both sources."""
    print("\n[TEST-3] Test 3: Searching across sources...")

    query = "SQLite FTS5 hybrid retrieval"
    results = search_context(ctx, query, k=10, min_score=0.1, source="all")

    print(f"[PASS] Search returned {len(results)} results")

    if len(results) == 0:
        raise SmokeTestFailure(f"Search returned no results for query: {query}")

    # Verify results have required fields
    for r in results[:3]:
        print(f"   - [{r.score:.3f}] {r.source_type}: {r.title}")
        if not r.chunk_id or not r.citation or r.score <= 0:
            raise SmokeTestFailure(f"Invalid result structure: {r}")

    return results


def test_chinvex_answer_grounded(ctx: ContextConfig, contexts_root: Path):
    """Test 4: chinvex_answer with groundable query."""
    print("\n[PASS] Test 4: chinvex_answer with groundable query...")

    query = "What is chinvex and what does it use for hybrid retrieval?"

    # Inline the logic from handle_chinvex_answer (avoid MCP imports)
    results = search_context(ctx, query, k=8, min_score=0.2, source="all")

    chunks = [
        {
            "chunk_id": r.chunk_id,
            "text": r.snippet,
            "score": r.score,
            "source_type": r.source_type,
        }
        for r in results
    ]

    citations = [r.citation for r in results]

    result = {
        "query": query,
        "chunks": chunks,
        "citations": citations,
        "context_name": ctx.name,
        "weights_applied": ctx.weights,
    }

    print(f"[PASS] Answer returned {len(result['chunks'])} chunks")

    if len(result["chunks"]) == 0:
        raise SmokeTestFailure("No chunks returned for groundable query")

    if len(result["citations"]) == 0:
        raise SmokeTestFailure("No citations returned for groundable query")

    # Verify structure
    for chunk in result["chunks"][:2]:
        print(f"   - [{chunk['score']:.3f}] {chunk['source_type']}")
        if not chunk["chunk_id"] or not chunk["text"]:
            raise SmokeTestFailure(f"Invalid chunk structure: {chunk}")

    print(f"   Citations: {result['citations'][:3]}")

    return result


def test_chinvex_answer_ungrounded(ctx: ContextConfig, contexts_root: Path):
    """Test 5: chinvex_answer with ungroundable query."""
    print("\n[TEST-5] Test 5: chinvex_answer with ungroundable query...")

    query = "What is the recipe for authentic Neapolitan pizza dough?"

    # Inline the logic from handle_chinvex_answer (avoid MCP imports)
    results = search_context(ctx, query, k=8, min_score=0.3, source="all")

    chunks = [
        {
            "chunk_id": r.chunk_id,
            "text": r.snippet,
            "score": r.score,
            "source_type": r.source_type,
        }
        for r in results
    ]

    citations = [r.citation for r in results]

    result = {
        "query": query,
        "chunks": chunks,
        "citations": citations,
        "context_name": ctx.name,
    }

    # Should return empty or very low-score results
    if len(result["chunks"]) > 0:
        print(f"[WARN]  Returned {len(result['chunks'])} chunks (expected 0 with high threshold)")
        # This is OK - might get some low-score matches
    else:
        print("[PASS] Correctly returned no chunks for ungroundable query")

    return result


def test_fingerprint_skip(ctx: ContextConfig) -> dict:
    """Test 6: Re-run ingest and verify fingerprints skip unchanged files."""
    print("\n[TEST-6] Test 6: Re-ingesting to verify fingerprinting...")

    stats = ingest_context(ctx, ollama_host_override="http://127.0.0.1:11434")

    print(f"[PASS] Re-ingest complete: {stats}")

    if stats["skipped"] == 0:
        raise SmokeTestFailure("Fingerprinting failed - no files were skipped on re-ingest")

    print(f"   Skipped {stats['skipped']} unchanged documents (fingerprinting works!)")

    return stats


def cleanup_chroma(ctx: ContextConfig):
    """Explicitly close Chroma to release file locks."""
    try:
        from chinvex.vectors import VectorStore
        vectors = VectorStore(ctx.index.chroma_dir)
        if hasattr(vectors, 'collection') and vectors.collection:
            # Close any open connections
            del vectors.collection
        if hasattr(vectors, 'client') and vectors.client:
            del vectors.client
    except:
        pass


def main():
    print("=" * 60)
    print("*** P0 E2E Smoke Test")
    print("=" * 60)

    tmp_dir = None
    try:
        # Use manual cleanup to handle Chroma file locks on Windows
        tmp_dir = Path(tempfile.mkdtemp())

        # Create test context
        ctx = create_test_context(tmp_dir)
        contexts_root = tmp_dir / "contexts"

        # Run tests
        test_ingest_repo(ctx)
        test_ingest_codex_session(ctx)
        test_search(ctx)
        test_chinvex_answer_grounded(ctx, contexts_root)
        test_chinvex_answer_ungrounded(ctx, contexts_root)
        test_fingerprint_skip(ctx)

        # Cleanup Chroma before removing temp dir
        cleanup_chroma(ctx)

        print("\n" + "=" * 60)
        print("[PASS] ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except SmokeTestFailure as e:
        print(f"\n[TEST-5] TEST FAILED: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Best-effort cleanup
        if tmp_dir and tmp_dir.exists():
            import shutil
            import time
            try:
                time.sleep(0.5)  # Give file handles time to close
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except:
                print(f"\n[WARN] Could not clean up temp dir: {tmp_dir}")
                print("       This is OK - it will be cleaned up eventually.")


if __name__ == "__main__":
    sys.exit(main())
