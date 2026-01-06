from pathlib import Path

from chinvex.storage import Storage


def test_fts5_available(tmp_path: Path) -> None:
    db_path = tmp_path / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()
