"""Test gateway multi-context API validation."""
import pytest
from pydantic import ValidationError
from chinvex.gateway.validation import MultiContextSearchRequest


def test_search_request_accepts_contexts_array():
    """Test MultiContextSearchRequest accepts contexts array."""
    req = MultiContextSearchRequest(
        contexts=["Chinvex", "Personal"],
        query="test",
        k=5
    )
    assert req.contexts == ["Chinvex", "Personal"]
    assert req.query == "test"


def test_search_request_accepts_contexts_all():
    """Test MultiContextSearchRequest accepts contexts='all'."""
    req = MultiContextSearchRequest(
        contexts="all",
        query="test",
        k=5
    )
    assert req.contexts == "all"


def test_search_request_accepts_single_context():
    """Test MultiContextSearchRequest accepts single context."""
    req = MultiContextSearchRequest(
        context="Chinvex",
        query="test",
        k=5
    )
    assert req.context == "Chinvex"


def test_search_request_rejects_both_context_and_contexts():
    """Test MultiContextSearchRequest rejects both context and contexts."""
    with pytest.raises(ValidationError):
        MultiContextSearchRequest(
            context="Chinvex",
            contexts=["Personal"],
            query="test"
        )
