"""Test archive filtering in search."""
import pytest
from chinvex.search import search
import inspect


def test_search_has_include_archive_parameter():
    """Test that search() has include_archive parameter."""
    sig = inspect.signature(search)

    # This test will fail until include_archive parameter is added
    assert "include_archive" in sig.parameters, "search() must have include_archive parameter"
