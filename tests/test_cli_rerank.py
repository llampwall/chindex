from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner
from chinvex.cli import app


runner = CliRunner()


def test_search_command_with_rerank_flag():
    """Test that --rerank flag is passed to search_context."""
    with patch("chinvex.search.search_context") as mock_search:
        mock_search.return_value = [
            MagicMock(chunk_id="c1", score=0.95, source_type="repo", title="test", citation="test.py", snippet="test")
        ]
        with patch("chinvex.context.load_context") as mock_load:
            mock_ctx = MagicMock()
            mock_ctx.name = "Chinvex"
            mock_load.return_value = mock_ctx

            result = runner.invoke(app, ["search", "--context", "Chinvex", "--rerank", "test query"])

            assert result.exit_code == 0
            # Verify rerank=True was passed
            call_args = mock_search.call_args
            assert call_args[1]["rerank"] is True


def test_search_command_without_rerank_flag():
    """Test that rerank defaults to False when flag not provided."""
    with patch("chinvex.search.search_context") as mock_search:
        mock_search.return_value = [
            MagicMock(chunk_id="c1", score=0.95, source_type="repo", title="test", citation="test.py", snippet="test")
        ]
        with patch("chinvex.context.load_context") as mock_load:
            mock_ctx = MagicMock()
            mock_ctx.name = "Chinvex"
            mock_load.return_value = mock_ctx

            result = runner.invoke(app, ["search", "--context", "Chinvex", "test query"])

            assert result.exit_code == 0
            # Verify rerank=False was passed (default)
            call_args = mock_search.call_args
            assert call_args[1]["rerank"] is False


def test_multi_context_search_with_rerank_flag():
    """Test that --rerank flag works with multi-context search."""
    with patch("chinvex.search.search_multi_context") as mock_search:
        mock_search.return_value = [
            MagicMock(chunk_id="c1", score=0.95, source_type="repo", title="test", citation="test.py", snippet="test", context="Chinvex")
        ]

        result = runner.invoke(app, ["search", "--contexts", "Chinvex,Codex", "--rerank", "test query"])

        assert result.exit_code == 0
        # Verify rerank=True was passed
        call_args = mock_search.call_args
        assert call_args[1]["rerank"] is True
