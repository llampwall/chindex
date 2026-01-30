# tests/sync/test_discovery.py
import pytest
import json
from pathlib import Path
from chinvex.sync.discovery import discover_watch_sources, WatchSource


def test_discover_empty_contexts(tmp_path: Path):
    """Empty contexts directory should return no sources"""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 0


def test_discover_single_repo_context(tmp_path: Path):
    """Should discover repo sources from context.json"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    # Create minimal context.json with repo
    ctx_config = {
        "name": "TestCtx",
        "includes": {
            "repos": [str(tmp_path / "repo1")]
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 1
    assert sources[0].context_name == "TestCtx"
    assert sources[0].source_type == "repo"
    assert sources[0].path == tmp_path / "repo1"


def test_discover_multiple_repos_same_context(tmp_path: Path):
    """Context with multiple repos should yield multiple sources"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    ctx_config = {
        "name": "TestCtx",
        "includes": {
            "repos": [
                str(tmp_path / "repo1"),
                str(tmp_path / "repo2")
            ]
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 2
    assert all(s.context_name == "TestCtx" for s in sources)
    assert {s.path for s in sources} == {tmp_path / "repo1", tmp_path / "repo2"}


def test_discover_inbox_source(tmp_path: Path):
    """Should discover inbox sources"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "_global"
    ctx_dir.mkdir(parents=True)

    ctx_config = {
        "name": "_global",
        "includes": {
            "inbox": [str(tmp_path / "inbox")]
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 1
    assert sources[0].source_type == "inbox"
    assert sources[0].path == tmp_path / "inbox"


def test_discover_skips_malformed_context(tmp_path: Path):
    """Malformed context.json should be skipped with warning"""
    contexts_root = tmp_path / "contexts"

    # Valid context
    ctx1 = contexts_root / "Ctx1"
    ctx1.mkdir(parents=True)
    (ctx1 / "context.json").write_text(json.dumps({
        "name": "Ctx1",
        "includes": {"repos": [str(tmp_path / "repo1")]}
    }))

    # Invalid context (malformed JSON)
    ctx2 = contexts_root / "Ctx2"
    ctx2.mkdir(parents=True)
    (ctx2 / "context.json").write_text("{ invalid json")

    sources = discover_watch_sources(contexts_root)

    # Should only find the valid one
    assert len(sources) == 1
    assert sources[0].context_name == "Ctx1"


def test_discover_multiple_contexts(tmp_path: Path):
    """Should discover sources from all contexts"""
    contexts_root = tmp_path / "contexts"

    for i in range(3):
        ctx_dir = contexts_root / f"Ctx{i}"
        ctx_dir.mkdir(parents=True)
        (ctx_dir / "context.json").write_text(json.dumps({
            "name": f"Ctx{i}",
            "includes": {"repos": [str(tmp_path / f"repo{i}")]}
        }))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 3
    assert {s.context_name for s in sources} == {"Ctx0", "Ctx1", "Ctx2"}
