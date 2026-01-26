from pathlib import Path
import json
import pytest
from chinvex.context import ContextConfig, load_context, list_contexts, ContextNotFoundError


def test_context_config_from_dict() -> None:
    data = {
        "schema_version": 1,
        "name": "Chinvex",
        "aliases": ["chindex"],
        "includes": {
            "repos": ["C:\\Code\\chinvex"],
            "chat_roots": ["P:\\ai_memory\\chats"],
            "codex_session_roots": ["C:\\Users\\Jordan\\.codex\\sessions"],
            "note_roots": ["P:\\ai_memory\\notes"]
        },
        "index": {
            "sqlite_path": "P:\\ai_memory\\indexes\\Chinvex\\hybrid.db",
            "chroma_dir": "P:\\ai_memory\\indexes\\Chinvex\\chroma"
        },
        "weights": {
            "repo": 1.0,
            "chat": 0.8,
            "codex_session": 0.9,
            "note": 0.7
        },
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }

    ctx = ContextConfig.from_dict(data)
    assert ctx.name == "Chinvex"
    assert "chindex" in ctx.aliases
    assert len(ctx.includes.repos) == 1
    assert ctx.weights["repo"] == 1.0


def test_load_context_by_name(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir()

    context_json = {
        "schema_version": 1,
        "name": "TestCtx",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {
            "sqlite_path": str(tmp_path / "indexes" / "TestCtx" / "hybrid.db"),
            "chroma_dir": str(tmp_path / "indexes" / "TestCtx" / "chroma")
        },
        "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }
    (ctx_dir / "context.json").write_text(json.dumps(context_json), encoding="utf-8")

    ctx = load_context("TestCtx", contexts_root)
    assert ctx.name == "TestCtx"


def test_load_context_by_alias(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    ctx_dir = contexts_root / "RealName"
    ctx_dir.mkdir()

    context_json = {
        "schema_version": 1,
        "name": "RealName",
        "aliases": ["shortname", "alias2"],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {
            "sqlite_path": str(tmp_path / "indexes" / "RealName" / "hybrid.db"),
            "chroma_dir": str(tmp_path / "indexes" / "RealName" / "chroma")
        },
        "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }
    (ctx_dir / "context.json").write_text(json.dumps(context_json), encoding="utf-8")

    ctx = load_context("shortname", contexts_root)
    assert ctx.name == "RealName"


def test_load_context_unknown_errors(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    with pytest.raises(ContextNotFoundError, match="Unknown context: NoSuchContext"):
        load_context("NoSuchContext", contexts_root)


def test_list_contexts(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    for name in ["Alpha", "Beta"]:
        ctx_dir = contexts_root / name
        ctx_dir.mkdir()
        context_json = {
            "schema_version": 1,
            "name": name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {
                "sqlite_path": str(tmp_path / "indexes" / name / "hybrid.db"),
                "chroma_dir": str(tmp_path / "indexes" / name / "chroma")
            },
            "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
            "created_at": "2026-01-26T00:00:00Z",
            "updated_at": "2026-01-26T12:00:00Z" if name == "Beta" else "2026-01-26T08:00:00Z"
        }
        (ctx_dir / "context.json").write_text(json.dumps(context_json), encoding="utf-8")

    contexts = list_contexts(contexts_root)
    assert len(contexts) == 2
    # Should be sorted by updated_at desc
    assert contexts[0].name == "Beta"
    assert contexts[1].name == "Alpha"
