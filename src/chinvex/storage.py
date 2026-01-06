from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .util import now_iso


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        self._check_fts5()
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents(
              doc_id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              project TEXT,
              repo TEXT,
              title TEXT,
              updated_at TEXT,
              content_hash TEXT,
              meta_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
              chunk_id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              source_type TEXT NOT NULL,
              project TEXT,
              repo TEXT,
              ordinal INTEGER NOT NULL,
              text TEXT NOT NULL,
              updated_at TEXT,
              meta_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs(
              run_id TEXT PRIMARY KEY,
              started_at TEXT,
              finished_at TEXT,
              stats_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, content='', tokenize='unicode61')
            """
        )
        self.conn.commit()

    def _check_fts5(self) -> None:
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(text)")
            self.conn.execute("DROP TABLE IF EXISTS _fts5_test")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "FTS5 not available in this Python/SQLite build. "
                "Install a Python build with SQLite FTS5 enabled."
            ) from exc

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.fetchone()

    def upsert_document(
        self,
        *,
        doc_id: str,
        source_type: str,
        source_uri: str,
        project: str | None,
        repo: str | None,
        title: str | None,
        updated_at: str,
        content_hash: str,
        meta_json: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO documents(doc_id, source_type, source_uri, project, repo, title, updated_at, content_hash, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
              source_type=excluded.source_type,
              source_uri=excluded.source_uri,
              project=excluded.project,
              repo=excluded.repo,
              title=excluded.title,
              updated_at=excluded.updated_at,
              content_hash=excluded.content_hash,
              meta_json=excluded.meta_json
            """,
            (doc_id, source_type, source_uri, project, repo, title, updated_at, content_hash, meta_json),
        )
        self.conn.commit()

    def delete_chunks_for_doc(self, doc_id: str) -> list[str]:
        cur = self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        chunk_ids = [row["chunk_id"] for row in cur.fetchall()]
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return chunk_ids

    def upsert_chunks(self, rows: Iterable[tuple]) -> None:
        self.conn.executemany(
            """
            INSERT INTO chunks(chunk_id, doc_id, source_type, project, repo, ordinal, text, updated_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              doc_id=excluded.doc_id,
              source_type=excluded.source_type,
              project=excluded.project,
              repo=excluded.repo,
              ordinal=excluded.ordinal,
              text=excluded.text,
              updated_at=excluded.updated_at,
              meta_json=excluded.meta_json
            """,
            rows,
        )
        self.conn.commit()

    def upsert_fts(self, rows: Iterable[tuple]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks_fts(rowid, text) VALUES ((SELECT rowid FROM chunks WHERE chunk_id = ?), ?)",
            rows,
        )
        self.conn.commit()

    def search_fts(self, query: str, limit: int = 30, filters: dict | None = None) -> list[sqlite3.Row]:
        filters = filters or {}
        clauses = []
        params: list = []
        if filters.get("source_type"):
            clauses.append("chunks.source_type = ?")
            params.append(filters["source_type"])
        if filters.get("project"):
            clauses.append("chunks.project = ?")
            params.append(filters["project"])
        if filters.get("repo"):
            clauses.append("chunks.repo = ?")
            params.append(filters["repo"])
        sql = f"""
            SELECT chunks.chunk_id, chunks.text, chunks.source_type, chunks.project, chunks.repo,
                   chunks.doc_id, chunks.ordinal, chunks.updated_at, chunks.meta_json,
                   bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks ON chunks_fts.rowid = chunks.rowid
            WHERE chunks_fts MATCH ?
            {('AND ' + ' AND '.join(clauses)) if clauses else ''}
            ORDER BY rank
            LIMIT ?
        """
        params = [query] + params + [limit]
        cur = self.conn.execute(sql, params)
        return cur.fetchall()

    def record_run(self, run_id: str, started_at: str, stats_json: str) -> None:
        finished = now_iso()
        self.conn.execute(
            """
            INSERT INTO ingestion_runs(run_id, started_at, finished_at, stats_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, started_at, finished, stats_json),
        )
        self.conn.commit()
