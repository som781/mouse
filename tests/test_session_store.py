"""Tests for the SQLite-backed SessionStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.sessions.storage import (
    SessionRecord,
    SessionStore,
    SessionStoreError,
)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    s = SessionStore(tmp_path / "db" / "sessions.db")
    yield s
    s.close()


# ─── Errors & construction ──────────────────────────────────────────


def test_error_is_mouse_error():
    assert issubclass(SessionStoreError, MouseError)


def test_store_creates_parent_directory(tmp_path: Path):
    db = tmp_path / "nested" / "a" / "sessions.db"
    assert not db.parent.exists()
    s = SessionStore(db)
    try:
        assert db.exists()
        assert db.parent.is_dir()
    finally:
        s.close()


def test_schema_is_idempotent(tmp_path: Path):
    db = tmp_path / "sessions.db"
    SessionStore(db).close()
    # Reopening on the same file must not error.
    s = SessionStore(db)
    try:
        assert s.list_sessions() == []
    finally:
        s.close()


# ─── Sessions CRUD ──────────────────────────────────────────────────


def test_create_and_get_session(store: SessionStore):
    rec = store.create_session(project_root="/tmp/proj", model="gpt-4")
    assert isinstance(rec, SessionRecord)
    assert len(rec.id) == 12
    assert rec.status == "active"
    assert rec.created_at > 0
    assert rec.updated_at == rec.created_at
    assert rec.total_tokens == 0
    assert rec.tags == []

    fetched = store.get_session(rec.id)
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.project_root == "/tmp/proj"
    assert fetched.model == "gpt-4"


def test_get_unknown_session_returns_none(store: SessionStore):
    assert store.get_session("nope") is None


def test_create_with_explicit_id_and_tags(store: SessionStore):
    rec = store.create_session(
        session_id="abc123",
        title="initial",
        tags=["bug", "urgent"],
    )
    assert rec.id == "abc123"
    assert rec.title == "initial"
    assert rec.tags == ["bug", "urgent"]


def test_list_sessions_orders_by_updated_at_desc(store: SessionStore):
    a = store.create_session(title="A")
    b = store.create_session(title="B")
    c = store.create_session(title="C")
    # Bump A so it becomes most-recent.
    store.update_session(a.id, title="A2")
    listed = store.list_sessions()
    assert [s.id for s in listed][:3] == [a.id, c.id, b.id]


def test_list_sessions_filters_by_status(store: SessionStore):
    a = store.create_session(title="A")
    b = store.create_session(title="B")
    store.update_session(b.id, status="completed")
    active = store.list_sessions(status="active")
    assert [s.id for s in active] == [a.id]
    done = store.list_sessions(status="completed")
    assert [s.id for s in done] == [b.id]


def test_update_session_bumps_updated_at(store: SessionStore):
    rec = store.create_session(title="old")
    import time as _t
    _t.sleep(0.01)
    store.update_session(rec.id, title="new", summary="did stuff")
    refreshed = store.get_session(rec.id)
    assert refreshed.title == "new"
    assert refreshed.summary == "did stuff"
    assert refreshed.updated_at > rec.updated_at


def test_update_rejects_unknown_field(store: SessionStore):
    rec = store.create_session()
    with pytest.raises(SessionStoreError, match="unknown"):
        store.update_session(rec.id, bogus="x")


def test_update_tags_must_be_list(store: SessionStore):
    rec = store.create_session()
    with pytest.raises(SessionStoreError, match="tags"):
        store.update_session(rec.id, tags="not-a-list")


def test_delete_session_cascades_messages(store: SessionStore):
    rec = store.create_session()
    store.append_message(rec.id, role="user", content="hi")
    store.append_message(rec.id, role="assistant", content="hello")
    assert store.message_count(rec.id) == 2
    store.delete_session(rec.id)
    assert store.get_session(rec.id) is None
    assert store.message_count(rec.id) == 0


# ─── Messages ───────────────────────────────────────────────────────


def test_append_and_fetch_messages(store: SessionStore):
    rec = store.create_session()
    store.append_message(rec.id, role="user", content="hello")
    store.append_message(
        rec.id,
        role="tool",
        content="result blob",
        tool_name="bash",
        tool_input='{"cmd":"ls"}',
        tool_output="file1\nfile2",
    )
    msgs = store.get_messages(rec.id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["tool_name"] == "bash"
    assert msgs[1]["tool_output"] == "file1\nfile2"


def test_append_message_bumps_session_updated_at(store: SessionStore):
    rec = store.create_session()
    import time as _t
    _t.sleep(0.01)
    store.append_message(rec.id, role="user", content="ping")
    refreshed = store.get_session(rec.id)
    assert refreshed.updated_at > rec.updated_at


# ─── Search ─────────────────────────────────────────────────────────


def test_search_empty_query_returns_empty(store: SessionStore):
    store.create_session(title="foo")
    assert store.search("") == []
    assert store.search("   ") == []


def test_search_finds_by_title(store: SessionStore):
    a = store.create_session(title="auth middleware fix")
    store.create_session(title="unrelated ticket")
    hits = store.search("auth")
    assert [s.id for s in hits] == [a.id]


def test_search_finds_by_summary(store: SessionStore):
    rec = store.create_session()
    store.update_session(rec.id, summary="investigated CORS error in FastAPI")
    hits = store.search("CORS")
    assert any(s.id == rec.id for s in hits)


def test_search_updates_reflect_in_fts(store: SessionStore):
    """After renaming a session, the old term must NOT match and the
    new term MUST match. This exercises the update trigger."""
    rec = store.create_session(title="first name")
    assert any(s.id == rec.id for s in store.search("first"))
    store.update_session(rec.id, title="renamed version")
    assert not any(s.id == rec.id for s in store.search("first"))
    assert any(s.id == rec.id for s in store.search("renamed"))


def test_search_malformed_fts_query_falls_back(store: SessionStore):
    """An unbalanced quote is an invalid FTS5 expression; store should
    fall back to LIKE and still return (a possibly empty) result without
    raising."""
    store.create_session(title='he said "hello')
    # Should not raise:
    result = store.search('"unterminated')
    assert isinstance(result, list)


def test_list_sessions_limit(store: SessionStore):
    for i in range(5):
        store.create_session(title=f"s{i}")
    assert len(store.list_sessions(limit=3)) == 3


def test_fts_enabled_property_is_true_on_modern_sqlite(store: SessionStore):
    # Every Python ≥ 3.11 we ship on has FTS5 compiled in.
    assert store.fts_enabled is True


def test_store_context_manager_closes(tmp_path: Path):
    with SessionStore(tmp_path / "cm.db") as s:
        s.create_session(title="inside")
    # After exit, a new instance can reopen the file and see the data.
    s2 = SessionStore(tmp_path / "cm.db")
    try:
        assert len(s2.list_sessions()) == 1
    finally:
        s2.close()


def test_update_session_with_no_fields_is_noop(store: SessionStore):
    rec = store.create_session(title="unchanged")
    store.update_session(rec.id)  # no kwargs
    refreshed = store.get_session(rec.id)
    assert refreshed.title == "unchanged"
    # updated_at is unchanged because we early-returned.
    assert refreshed.updated_at == rec.updated_at


def test_update_session_replaces_tags(store: SessionStore):
    rec = store.create_session(tags=["old"])
    store.update_session(rec.id, tags=["new", "shiny"])
    refreshed = store.get_session(rec.id)
    assert refreshed.tags == ["new", "shiny"]
