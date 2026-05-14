"""Tests for progress.md handoff generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.sessions import SessionManager
from mouse.sessions.handoff import (
    HandoffError,
    MAX_TRANSCRIPT_CHARS,
    build_handoff_prompt,
    generate_progress,
    handle_progress,
    progress_path,
    save_progress,
)
from mouse.tools import tool_output_store

from tests.conftest import FakeMessage, FakeResponse


_GOOD_PROGRESS = """# Progress

## Completed
- [x] Set up auth middleware

## In Progress
- [ ] Wire up Stripe webhook

## Discovered
- Rate limiting needed on /api/register

## Files Modified
- src/auth/middleware.py
"""


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
    def __init__(self, llm) -> None:
        self.llm = llm


# ─── Errors ──────────────────────────────────────────────────────────


def test_handoff_error_is_mouse_error():
    assert issubclass(HandoffError, MouseError)


# ─── Prompt assembly ────────────────────────────────────────────────


def test_build_prompt_includes_user_and_assistant_and_tool():
    msgs = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "looking into it"},
        {"role": "tool", "tool_name": "bash",
         "tool_input": '{"command": "ls"}',
         "tool_output": "file.py"},
    ]
    prompt = build_handoff_prompt(msgs)
    assert "USER: fix the bug" in prompt
    assert "ASSISTANT: looking into it" in prompt
    assert "TOOL[bash]" in prompt
    assert "file.py" in prompt


def test_build_prompt_truncates_long_tool_output():
    big = "x" * 2000
    msgs = [{"role": "tool", "tool_name": "bash",
             "tool_input": "{}", "tool_output": big}]
    prompt = build_handoff_prompt(msgs)
    assert "truncated" in prompt
    # Should not contain the full 2000-char blob.
    assert "x" * 2000 not in prompt


def test_build_prompt_trims_long_transcript_keeping_tail():
    # 100 messages × 300 chars each = way over the cap.
    msgs = [{"role": "user", "content": "u" * 300} for _ in range(100)]
    msgs.append({"role": "user", "content": "TAIL MARKER"})
    prompt = build_handoff_prompt(msgs)
    assert "older context trimmed" in prompt
    assert "TAIL MARKER" in prompt
    # The total size, minus the fixed prompt preamble, is bounded.
    assert len(prompt) < MAX_TRANSCRIPT_CHARS + 500


def test_build_prompt_skips_empty_messages():
    msgs = [
        {"role": "user", "content": "real"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "   "},
    ]
    prompt = build_handoff_prompt(msgs)
    assert "USER: real" in prompt
    # Only one non-empty line in the transcript section.
    body = prompt.split("\n\n", 1)[-1]
    assert body.count("ASSISTANT:") == 0


# ─── Generation ─────────────────────────────────────────────────────


def test_generate_progress_returns_llm_output(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content=_GOOD_PROGRESS)))
    out = generate_progress(
        [{"role": "user", "content": "hi"}], llm,
    )
    assert out.startswith("# Progress")
    assert "Set up auth middleware" in out
    # The call actually consulted the LLM.
    assert len(llm.calls) == 1


def test_generate_progress_empty_messages_returns_skeleton(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content="x")))
    out = generate_progress([], llm)
    assert "# Progress" in out
    assert "_None yet._" in out
    # No LLM call for empty input.
    assert llm.calls == []


def test_generate_progress_falls_back_on_llm_error():
    class BrokenLLM:
        def complete(self, **kwargs):
            raise RuntimeError("boom")

    out = generate_progress(
        [{"role": "user", "content": "hi"}], BrokenLLM(),
    )
    assert "# Progress" in out
    assert "_None yet._" in out


def test_generate_progress_falls_back_on_empty_response(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content="")))
    out = generate_progress(
        [{"role": "user", "content": "hi"}], llm,
    )
    assert "_None yet._" in out


def test_generate_progress_prepends_heading_when_missing(make_llm):
    """Defensive: the LLM sometimes skips the top-level heading."""
    llm = make_llm(FakeResponse(FakeMessage(content="## Completed\n- [x] thing")))
    out = generate_progress([{"role": "user", "content": "hi"}], llm)
    assert out.startswith("# Progress")
    assert "## Completed" in out


# ─── Persistence ────────────────────────────────────────────────────


def test_progress_path_matches_session_layout(tmp_path: Path):
    p = progress_path(tmp_path / "mouse", "abc123")
    assert p == tmp_path / "mouse" / "sessions" / "abc123" / "progress.md"


def test_save_progress_writes_file(mgr: SessionManager):
    rec = mgr.create()
    path = save_progress(mgr, _GOOD_PROGRESS)
    assert path.exists()
    assert path.read_text() == _GOOD_PROGRESS
    assert path == progress_path(mgr.base_dir, rec.id)


def test_save_progress_without_active_raises(mgr: SessionManager):
    with pytest.raises(HandoffError, match="no active session"):
        save_progress(mgr, _GOOD_PROGRESS)


def test_save_progress_creates_parent_dirs(mgr: SessionManager):
    rec = mgr.create()
    # Make sure nothing is pre-created under the session dir.
    assert not (mgr.base_dir / "sessions" / rec.id).exists()
    save_progress(mgr, _GOOD_PROGRESS)
    assert (mgr.base_dir / "sessions" / rec.id / "progress.md").exists()


# ─── CLI handler ────────────────────────────────────────────────────


def test_handle_progress_no_active_session(mgr: SessionManager, make_llm):
    agent = FakeAgent(make_llm(FakeResponse(FakeMessage(content=_GOOD_PROGRESS))))
    out = handle_progress(mgr, agent)
    assert "No active session" in out


def test_handle_progress_no_messages(mgr: SessionManager, make_llm):
    mgr.create()
    agent = FakeAgent(make_llm(FakeResponse(FakeMessage(content=_GOOD_PROGRESS))))
    out = handle_progress(mgr, agent)
    assert "nothing to hand off" in out


def test_handle_progress_writes_file(mgr: SessionManager, make_llm):
    rec = mgr.create()
    mgr.record_user("fix auth")
    mgr.record_assistant("done")
    agent = FakeAgent(make_llm(FakeResponse(FakeMessage(content=_GOOD_PROGRESS))))
    out = handle_progress(mgr, agent)

    assert "Wrote progress.md" in out
    path = progress_path(mgr.base_dir, rec.id)
    assert path.exists()
    assert "Set up auth middleware" in path.read_text()


def test_handle_progress_reports_agent_without_llm(mgr: SessionManager):
    mgr.create()
    mgr.record_user("x")

    class NoLLM:
        pass
    out = handle_progress(mgr, NoLLM())
    assert "no llm" in out.lower()
