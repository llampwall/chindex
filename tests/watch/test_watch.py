# tests/watch/test_watch.py
from chinvex.watch.models import Watch

def test_watch_creation():
    """Test Watch model creation."""
    watch = Watch(
        id="test_watch",
        query="test query",
        min_score=0.75,
        enabled=True,
        created_at="2026-01-26T00:00:00Z"
    )
    assert watch.id == "test_watch"
    assert watch.enabled is True
