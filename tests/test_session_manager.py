"""Tests for SessionManager — the storage + transcript + tool-output façade."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.sessions import (
    SessionManager,
    SessionManagerError,
    read_transcript,
)
from mouse.tools import tool_output_store

from tests.conftest import FakeMessage, FakeResponse


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    # Reset global tool-output store between tests.
    tool_output_store.configure(None)
    tool_output_store.clear()
    m = SessionManager(tmp_path / "mouse")
    yield m
    m.close()
    tool_output_store.configure(None)
    tool_output_store.clear()


# ─── Construction ───────────────────────────────────────────────────


def test_error_is_mouse_error():
    assert issubclass(SessionManagerError, MouseError)


def test_manager_creates_base_dir(tmp_path: Path):
    base = tmp_path / "fresh"
    assert not base.exists()
    m = SessionManager(base)
    try:
        assert base.is_dir()
        assert (base / "sessions.db").exists()
    finally:
        m.close()


# ─── Create / resume / end ──────────────────────────────────────────


def test_create_makes_session_active(manager: SessionManager):
    rec = manager.create(project_root="/p", model="fake")
    assert manager.active is not None
    assert manager.active.id == rec.id
    assert manager.active.project_root == "/p"
    assert manager.active.model == "fake"


def test_create_wires_tool_output_store_to_session_dir(manager: SessionManager):
    rec = manager.create()
    tool_output_store.save("call_xyz", "live payload")
    session_dir = manager.base_dir / "sessions" / rec.id
    assert (session_dir / "tool_outputs" / "call_xyz.txt").exists()


def test_end_detaches_tool_output_store(manager: SessionManager):
    manager.create()
    tool_output_store.save("a", "one")
    manager.end()
    # After end(), the store falls back to memory and sees nothing from
    # the ended session.
    assert tool_output_store.size() == 0


def test_create_while_active_ends_previous(manager: SessionManager):
    first = manager.create(title="A")
    manager.record_user("hello from A")
    second = manager.create(title="B")
    assert manager.active.id == second.id
    assert first.id != second.id
    # The previous session's messages were persisted before we rolled.
    msgs_a = manager.store.get_messages(first.id)
    assert [m["content"] for m in msgs_a] == ["hello from A"]


def test_resume_reattaches_active_session(manager: SessionManager):
    rec = manager.create(title="persist me")
    manager.record_user("u1")
    manager.end()
    assert manager.active is None
    resumed = manager.resume(rec.id)
    assert resumed.id == rec.id
    assert manager.active.id == rec.id


def test_resume_unknown_raises(manager: SessionManager):
    with pytest.raises(SessionManagerError, match="unknown session id"):
        manager.resume("nope")


def test_resume_noop_when_already_active(manager: SessionManager):
    rec = manager.create()
    again = manager.resume(rec.id)
    assert again.id == rec.id


# ─── Recording ──────────────────────────────────────────────────────


def test_record_user_and_assistant_land_in_store_and_transcript(manager: SessionManager):
    rec = manager.create()
    manager.record_user("what time is it")
    manager.record_assistant("about 3pm")

    msgs = manager.store.get_messages(rec.id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["content"] for m in msgs] == ["what time is it", "about 3pm"]

    # Transcript mirrors the same content.
    transcript_path = (
        manager.base_dir / "transcripts"
        / manager._active.writer.path.name  # noqa: SLF001 — internal for test
    )
    records = read_transcript(transcript_path)
    assert [r["role"] for r in records] == ["user", "assistant"]
    assert all("seq" in r for r in records)


def test_record_tool_captures_all_fields(manager: SessionManager):
    rec = manager.create()
    manager.record_tool(
        tool_name="bash",
        tool_input='{"command": "ls"}',
        tool_output="file.txt",
        tool_call_id="call_1",
    )
    msgs = manager.store.get_messages(rec.id)
    assert len(msgs) == 1
    assert msgs[0]["tool_name"] == "bash"
    assert msgs[0]["tool_output"] == "file.txt"
    assert msgs[0]["tool_input"] == '{"command": "ls"}'


def test_record_without_active_session_raises(manager: SessionManager):
    with pytest.raises(SessionManagerError, match="no active session"):
        manager.record_user("oops")


def test_update_token_count_and_set_title(manager: SessionManager):
    rec = manager.create()
    manager.update_token_count(123)
    manager.set_title("investigating auth bug")
    refreshed = manager.store.get_session(rec.id)
    assert refreshed.total_tokens == 123
    assert refreshed.title == "investigating auth bug"


def test_set_title_caps_length(manager: SessionManager):
    manager.create()
    long = "x" * 200
    manager.set_title(long)
    assert len(manager.active.title) == 60


def test_set_summary_persists_to_store(manager: SessionManager):
    rec = manager.create()
    manager.set_summary("refactored the auth middleware and added tests")
    refreshed = manager.store.get_session(rec.id)
    assert refreshed.summary == "refactored the auth middleware and added tests"
    assert manager.active.summary == refreshed.summary


def test_set_summary_requires_active_session(manager: SessionManager):
    with pytest.raises(SessionManagerError, match="no active session"):
        manager.set_summary("orphan")


# ─── Listing / searching / deleting ─────────────────────────────────


def test_list_returns_active_first(manager: SessionManager):
    a = manager.create(title="A")
    b = manager.create(title="B")  # makes B active, ends A
    listing = manager.list_sessions()
    ids = [s.id for s in listing]
    assert ids[0] == b.id
    assert a.id in ids


def test_search_proxies_to_store(manager: SessionManager):
    rec = manager.create(title="CORS middleware bug")
    manager.create(title="unrelated feature")
    hits = manager.search("CORS")
    assert any(s.id == rec.id for s in hits)


def test_delete_active_session_ends_it_first(manager: SessionManager):
    rec = manager.create()
    manager.record_user("hi")
    manager.delete(rec.id)
    assert manager.active is None
    assert manager.get(rec.id) is None


# ─── Auto-title ─────────────────────────────────────────────────────


def test_auto_title_uses_llm_response(manager: SessionManager, make_llm):
    manager.create()
    manager.record_user("I need help debugging a CORS failure in FastAPI")
    manager.record_assistant("Sure — can you share the error message?")
    llm = make_llm(FakeResponse(FakeMessage(content="Debug FastAPI CORS failure")))

    title = manager.auto_title(llm)
    assert title == "Debug FastAPI CORS failure"
    assert manager.active.title == title
    # The title prompt was sent with both first messages.
    sent = llm.calls[0]["messages"]
    assert any("CORS failure" in m["content"] for m in sent)


def test_auto_title_strips_quotes(manager: SessionManager, make_llm):
    manager.create()
    manager.record_user("hello")
    llm = make_llm(FakeResponse(FakeMessage(content='"Quoted title"')))
    title = manager.auto_title(llm)
    assert title == "Quoted title"


def test_auto_title_noop_when_title_exists(manager: SessionManager, make_llm):
    manager.create(title="already titled")
    manager.record_user("x")
    llm = make_llm(FakeResponse(FakeMessage(content="new title")))
    title = manager.auto_title(llm)
    assert title == "already titled"
    # LLM was not called.
    assert llm.calls == []


def test_auto_title_force_overrides_existing(manager: SessionManager, make_llm):
    manager.create(title="old")
    manager.record_user("explain decorators")
    llm = make_llm(FakeResponse(FakeMessage(content="Python decorators explained")))
    title = manager.auto_title(llm, force=True)
    assert title == "Python decorators explained"


def test_auto_title_falls_back_when_llm_raises(manager: SessionManager):
    class BrokenLLM:
        def complete(self, **kwargs):
            raise RuntimeError("upstream is down")

    manager.create()
    manager.record_user("how do I write a pytest fixture?")
    title = manager.auto_title(BrokenLLM())
    # Fallback is the first line of the user message, trimmed.
    assert title.startswith("how do I write a pytest fixture?")
    assert manager.active.title == title


def test_auto_title_without_messages_returns_empty(manager: SessionManager, make_llm):
    manager.create()
    llm = make_llm(FakeResponse(FakeMessage(content="irrelevant")))
    assert manager.auto_title(llm) == ""
    assert llm.calls == []  # never consulted


# ─── Context manager ────────────────────────────────────────────────


def test_context_manager_closes_everything(tmp_path: Path):
    with SessionManager(tmp_path / "cm") as m:
        rec = m.create()
        m.record_user("inside")
    # After exit, a fresh manager on the same dir sees the data.
    with SessionManager(tmp_path / "cm") as m2:
        refreshed = m2.get(rec.id)
        assert refreshed is not None
        msgs = m2.store.get_messages(rec.id)
        assert [msg["content"] for msg in msgs] == ["inside"]
