from dataclasses import dataclass, field

@dataclass(frozen=True)
class Watch:
    """Watch configuration for monitoring queries."""
    id: str
    query: str
    min_score: float
    enabled: bool
    created_at: str  # ISO8601

@dataclass(frozen=True)
class WatchConfig:
    """watch.json file schema."""
    schema_version: int
    watches: list[Watch] = field(default_factory=list)
