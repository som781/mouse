"""Tests for /new, /resume, /sessions, /search slash command handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.sessions import SessionManager
from mouse.sessions.commands import (
    RESUME_MESSAGE_LIMIT,
    _fmt_age,
    handle_new,
    handle_resume,
    handle_search,
    handle_sessions,
    handle_slash_session,
)
from mouse.tools import tool_output_store


@pytest.fixture
def mgr(tmp_path: Path) -> SessionManager:
    tool_output_store.configure(None)
    tool_output_store.clear()
    m = SessionManager(tmp_path / "mouse")
    yield m
    m.close()
    tool_output_store.configure(None)
    tool_output_store.clear()


class FakeAgent:
    """Just enough surface for the command handlers to poke at."""

    def __init__(self, model: str = "fake-model") -> None:
        self.messages: list[dict] = []
        self.reset_called = 0

        class _LLM:
            def __init__(self, m: str) -> None:
                self.model = m
        self.llm = _LLM(model)

    def reset(self) -> None:
        self.reset_called += 1
        self.messages = []


# ─── /new ────────────────────────────────────────────────────────────


def test_new_creates_session_and_resets_agent(mgr: SessionManager):
    agent = FakeAgent()
    out = handle_new(mgr, agent, title="auth bug")
    assert "Created session" in out
    assert "auth bug" in out
    assert mgr.active is not None
    assert mgr.active.title == "auth bug"
    assert agent.reset_called == 1


def test_new_captures_model_from_agent(mgr: SessionManager):
    agent = FakeAgent(model="claude-sonnet-4-6")
    handle_new(mgr, agent)
    assert mgr.active.model == "claude-sonnet-4-6"


def test_new_ends_previous_session(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent, title="A")
    first_id = mgr.active.id
    mgr.record_user("hello A")
    handle_new(mgr, agent, title="B")
    assert mgr.active.id != first_id
    assert mgr.active.title == "B"
    # Agent was reset again.
    assert agent.reset_called == 2


# ─── /resume ─────────────────────────────────────────────────────────


def test_resume_replays_user_and_assistant_messages(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent, title="keep me")
    mgr.record_user("original question")
    mgr.record_assistant("original answer")
    # Tool messages must NOT come back into agent.messages.
    mgr.record_tool(
        tool_name="bash", tool_input='{"command": "ls"}',
        tool_output="file.txt", tool_call_id="call_1",
    )
    rec_id = mgr.active.id
    mgr.end()
    agent.messages = [{"role": "user", "content": "stale"}]
    out = handle_resume(mgr, agent, rec_id)

    assert "Resumed session" in out
    assert "keep me" in out
    assert mgr.active.id == rec_id
    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant"]
    assert agent.messages[0]["content"] == "original question"


def test_resume_unknown_session_returns_error_message(mgr: SessionManager):
    agent = FakeAgent()
    out = handle_resume(mgr, agent, "does-not-exist")
    assert out.startswith("✗")
    assert "unknown session id" in out


def test_resume_caps_replay_to_limit(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent)
    # 60 user/assistant pairs (120 total) — way over the limit.
    for i in range(60):
        mgr.record_user(f"u{i}")
        mgr.record_assistant(f"a{i}")
    rec_id = mgr.active.id
    mgr.end()
    handle_resume(mgr, agent, rec_id)
    assert len(agent.messages) == RESUME_MESSAGE_LIMIT
    # The *last* messages should be the ones kept.
    assert agent.messages[-1]["content"] == "a59"


def test_resume_without_replay(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent)
    mgr.record_user("keep me")
    rec_id = mgr.active.id
    mgr.end()
    agent.messages = [{"role": "user", "content": "untouched"}]
    handle_resume(mgr, agent, rec_id, replay_into_agent=False)
    # Without replay, agent.messages should be left alone.
    assert agent.messages == [{"role": "user", "content": "untouched"}]


# ─── /sessions ───────────────────────────────────────────────────────


def test_sessions_empty(mgr: SessionManager):
    out = handle_sessions(mgr)
    assert "No sessions yet" in out


def test_sessions_lists_with_active_marker(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent, title="A")
    handle_new(mgr, agent, title="B")
    out = handle_sessions(mgr)
    assert "B" in out and "A" in out
    # Active session (B) is marked.
    assert "→" in out


def test_sessions_respects_limit(mgr: SessionManager):
    agent = FakeAgent()
    for i in range(5):
        handle_new(mgr, agent, title=f"S{i}")
    out = handle_sessions(mgr, limit=2)
    # Header line + exactly 2 entries.
    assert out.count("\n") == 2


# ─── /search ─────────────────────────────────────────────────────────


def test_search_empty_query_shows_usage(mgr: SessionManager):
    assert handle_search(mgr, "") == "Usage: /search <query>"
    assert handle_search(mgr, "   ") == "Usage: /search <query>"


def test_search_no_matches(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent, title="something else")
    out = handle_search(mgr, "xyzzy")
    assert "No sessions match" in out


def test_search_finds_by_title(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent, title="debug CORS error")
    handle_new(mgr, agent, title="unrelated task")
    out = handle_search(mgr, "CORS")
    assert "debug CORS error" in out


# ─── Dispatcher ──────────────────────────────────────────────────────


def test_dispatcher_routes_new(mgr: SessionManager):
    agent = FakeAgent()
    out = handle_slash_session("/new my task", mgr, agent)
    assert out is not None and "Created" in out
    assert mgr.active.title == "my task"


def test_dispatcher_returns_none_for_unknown_command(mgr: SessionManager):
    agent = FakeAgent()
    assert handle_slash_session("/help", mgr, agent) is None
    assert handle_slash_session("", mgr, agent) is None


def test_dispatcher_routes_resume_requires_id(mgr: SessionManager):
    agent = FakeAgent()
    out = handle_slash_session("/resume", mgr, agent)
    assert out is not None and "Usage" in out


def test_dispatcher_routes_resume_with_id(mgr: SessionManager):
    agent = FakeAgent()
    rec = mgr.create(title="target")
    mgr.record_user("anchor")
    mgr.end()
    out = handle_slash_session(f"/resume {rec.id}", mgr, agent)
    assert out is not None and "Resumed" in out
    assert mgr.active.id == rec.id
    assert agent.messages[0]["content"] == "anchor"


def test_dispatcher_routes_sessions_and_search(mgr: SessionManager):
    agent = FakeAgent()
    handle_new(mgr, agent, title="findme")
    assert "findme" in handle_slash_session("/sessions", mgr, agent)
    assert "findme" in handle_slash_session("/search findme", mgr, agent)


def test_dispatcher_is_case_insensitive_on_command(mgr: SessionManager):
    agent = FakeAgent()
    out = handle_slash_session("/NEW Upper", mgr, agent)
    assert out is not None and "Created" in out


# ─── _fmt_age ────────────────────────────────────────────────────────


def test_fmt_age_all_buckets():
    import time
    now = time.time()
    assert _fmt_age(now - 10).endswith("s ago")
    assert _fmt_age(now - 90).endswith("m ago")
    assert _fmt_age(now - 7200).endswith("h ago")
    assert _fmt_age(now - 2 * 86400).endswith("d ago")
    # Older than 7 days falls back to an ISO date.
    old = _fmt_age(now - 30 * 86400)
    assert len(old) == 10 and old[4] == "-" and old[7] == "-"
