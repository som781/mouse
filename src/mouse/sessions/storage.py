"""SQLite-backed session index with FTS5 full-text search.

The store is the authoritative index of sessions and their messages:
    ~/.mouse/sessions/sessions.db

Two regular tables and one FTS5 virtual table:

    sessions(id PK, title, created_at, updated_at, project_root, model,
             total_tokens, status, summary, tags)
    messages(id PK, session_id FK, role, content, timestamp,
             token_count, tool_name, tool_input, tool_output)
    session_fts(title, summary, tags)  -- mirrors sessions via triggers

Triggers keep ``session_fts`` in sync with ``sessions`` so callers just
update the regular table and search works for free. FTS5 is a SQLite
built-in on every modern Python ≥ 3.11, but we probe for it at open time
and fall back to a ``LIKE``-based search if it's missing (some minimal
Python builds strip FTS5). Tests cover the FTS5 path.

The JSONL transcript is the source of truth for replay; this index is
what ``/sessions``, ``/search``, and ``/resume`` read. Keeping them
separate means we can rebuild the index from transcripts if the DB is
ever lost or corrupted (Sprint 3 Step 2 builds the writer).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mouse.errors import MouseError


class SessionStoreError(MouseError):
    """Raised on unexpected storage-layer failures."""


# ─── Records ─────────────────────────────────────────────────────────


@dataclass
class SessionRecord:
    """A row from the ``sessions`` table."""

    id: str
    title: str
    created_at: float
    updated_at: float
    project_root: str
    model: str
    total_tokens: int
    status: str
    summary: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SessionRecord:
        raw_tags = row["tags"] or "[]"
        try:
            tags = json.loads(raw_tags)
            if not isinstance(tags, list):
                tags = []
        except json.JSONDecodeError:
            tags = []
        return cls(
            id=row["id"],
            title=row["title"] or "",
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            project_root=row["project_root"] or "",
            model=row["model"] or "",
            total_tokens=int(row["total_tokens"] or 0),
            status=row["status"] or "active",
            summary=row["summary"] or "",
            tags=tags,
        )


# ─── Schema ──────────────────────────────────────────────────────────


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT DEFAULT '',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    project_root  TEXT DEFAULT '',
    model         TEXT DEFAULT '',
    total_tokens  INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'active',
    summary       TEXT DEFAULT '',
    tags          TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
    ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role         TEXT NOT NULL,
    content      TEXT DEFAULT '',
    timestamp    REAL NOT NULL,
    token_count  INTEGER DEFAULT 0,
    tool_name    TEXT DEFAULT '',
    tool_input   TEXT DEFAULT '',
    tool_output  TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);
"""


_FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
    title, summary, tags, content=''
);

-- Mirror inserts/updates/deletes from sessions into session_fts.
-- content='' makes session_fts a contentless external-content table,
-- so we push the rowid explicitly.
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO session_fts(rowid, title, summary, tags)
    VALUES (new._rowid_, new.title, new.summary, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO session_fts(session_fts, rowid, title, summary, tags)
    VALUES ('delete', old._rowid_, old.title, old.summary, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO session_fts(session_fts, rowid, title, summary, tags)
    VALUES ('delete', old._rowid_, old.title, old.summary, old.tags);
    INSERT INTO session_fts(rowid, title, summary, tags)
    VALUES (new._rowid_, new.title, new.summary, new.tags);
END;
"""


_UPDATABLE_FIELDS = {
    "title", "summary", "status", "model",
    "project_root", "total_tokens", "tags",
}


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """Probe the sqlite build for FTS5 support."""
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _new_id() -> str:
    """Short, sortable-ish session id. Full uuid would overwhelm the CLI."""
    return uuid.uuid4().hex[:12]


# ─── Store ───────────────────────────────────────────────────────────


class SessionStore:
    """SQLite index for sessions and their messages.

    Thread-safety: the connection is opened with
    ``check_same_thread=False`` so the CLI (which stays on the main
    thread) and any background helpers can share it. All writes go
    through this single connection, and we serialise at the SQLite
    level, not at the Python level. Concurrent writers from separate
    processes are NOT supported — open one store per process.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._conn.executescript(_SCHEMA_SQL)
        self._fts_enabled = _fts5_available(self._conn)
        if self._fts_enabled:
            self._conn.executescript(_FTS_SCHEMA_SQL)

    # ── Lifecycle ──

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def fts_enabled(self) -> bool:
        """Whether the underlying sqlite build supports FTS5."""
        return self._fts_enabled

    # ── Sessions ──

    def create_session(
        self,
        *,
        session_id: str | None = None,
        project_root: str = "",
        model: str = "",
        title: str = "",
        tags: Iterable[str] | None = None,
    ) -> SessionRecord:
        """Insert a new session row and return the resulting record."""
        sid = session_id or _new_id()
        now = time.time()
        tags_json = json.dumps(list(tags or []))
        self._conn.execute(
            """
            INSERT INTO sessions (
                id, title, created_at, updated_at,
                project_root, model, total_tokens,
                status, summary, tags
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'active', '', ?)
            """,
            (sid, title, now, now, project_root, model, tags_json),
        )
        rec = self.get_session(sid)
        assert rec is not None  # we just inserted it
        return rec

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return SessionRecord.from_row(row) if row else None

    def list_sessions(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
    ) -> list[SessionRecord]:
        """Most-recently-updated first."""
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [SessionRecord.from_row(r) for r in rows]

    def update_session(self, session_id: str, **fields: Any) -> None:
        """Update whitelisted fields. ``updated_at`` is refreshed automatically."""
        if not fields:
            return
        cleaned: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in _UPDATABLE_FIELDS:
                raise SessionStoreError(
                    f"update_session: unknown or non-updatable field {key!r}"
                )
            if key == "tags":
                if not isinstance(value, (list, tuple)):
                    raise SessionStoreError("tags must be a list of strings")
                cleaned[key] = json.dumps(list(value))
            else:
                cleaned[key] = value
        cleaned["updated_at"] = time.time()

        set_clause = ", ".join(f"{k} = ?" for k in cleaned)
        params: list[Any] = list(cleaned.values())
        params.append(session_id)
        self._conn.execute(
            f"UPDATE sessions SET {set_clause} WHERE id = ?", params
        )

    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ── Messages ──

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str = "",
        token_count: int = 0,
        tool_name: str = "",
        tool_input: str = "",
        tool_output: str = "",
    ) -> int:
        """Append one message row. Returns the new row id.

        Also bumps ``sessions.updated_at`` so listings reflect activity.
        """
        ts = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO messages (
                session_id, role, content, timestamp,
                token_count, tool_name, tool_input, tool_output
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, ts, token_count,
             tool_name, tool_input, tool_output),
        )
        # Keep updated_at fresh without clobbering other fields.
        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (ts, session_id),
        )
        return int(cur.lastrowid)

    def get_messages(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def message_count(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    # ── Search ──

    def search(self, query: str, *, limit: int = 20) -> list[SessionRecord]:
        """Full-text search across title/summary/tags.

        Uses FTS5 when available; falls back to a case-insensitive LIKE
        scan otherwise. Returns ranked matches, most relevant first.
        """
        query = (query or "").strip()
        if not query:
            return []

        if self._fts_enabled:
            try:
                rows = self._conn.execute(
                    """
                    SELECT s.* FROM sessions s
                    JOIN session_fts f ON f.rowid = s._rowid_
                    WHERE session_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
                return [SessionRecord.from_row(r) for r in rows]
            except sqlite3.OperationalError:
                # Malformed FTS5 query (e.g. unbalanced quotes) — fall
                # through to LIKE so the caller still gets *something*.
                pass

        pattern = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT * FROM sessions
            WHERE title LIKE ? OR summary LIKE ? OR tags LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit),
        ).fetchall()
        return [SessionRecord.from_row(r) for r in rows]
