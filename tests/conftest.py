"""Global pytest fixtures and configuration for the chinvex test suite.

The key fixture here ensures the global SQLite connection (used by Storage) is
reset after every test. This prevents thread-safety errors that occur when:
  1. One test uses FastAPI TestClient which runs handlers in a worker thread,
     creating the global _CONN in that thread.
  2. A subsequent test tries to use Storage in the main test thread, but the
     global _CONN was created in the worker thread and SQLite raises:
     "SQLite objects created in a thread can only be used in that same thread."

The fix: directly null out the module-level _CONN and _CONN_PATH globals WITHOUT
calling _CONN.close(), since close() itself raises ProgrammingError when called
from a different thread than the one that created the connection. The leaked
connection will be garbage-collected by SQLite internally.
"""

import pytest


def _null_global_sqlite_connection() -> None:
    """Null out the global SQLite connection without closing it.

    Calling .close() on a cross-thread connection raises ProgrammingError.
    Instead, we abandon the connection (it will be GC'd) and reset the globals
    so the next Storage instantiation creates a fresh connection in the current
    thread.
    """
    try:
        import chinvex.storage as _storage_mod
        _storage_mod._CONN = None
        _storage_mod._CONN_PATH = None
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_sqlite_global_connection():
    """Null out the global SQLite connection before and after each test.

    This ensures each test gets a fresh connection in the current thread,
    avoiding cross-thread SQLite access errors when prior tests created
    connections in worker threads (e.g., FastAPI TestClient).

    We intentionally do NOT call _CONN.close() because that itself raises
    ProgrammingError when the connection was created in a different thread.
    """
    _null_global_sqlite_connection()
    yield
    _null_global_sqlite_connection()
