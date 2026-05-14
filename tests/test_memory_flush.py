"""Tests for the pre-compaction episodic flush hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.memory.episodic import Observation
from mouse.memory.flush import (
    classify,
    compact_with_flush,
    extract_observations,
    flush_to_episodic,
)
from mouse.memory.manager import MemoryManager
from mouse.sessions.compaction import CompactConfig

from tests.conftest import FakeMessage, FakeResponse


@pytest.fixture
def mem(tmp_path: Path) -> MemoryManager:
    return MemoryManager(tmp_path / "memory")


# ─── Classifier ─────────────────────────────────────────────────────


def test_classify_fixed():
    assert classify("I fixed the CORS bug") == "fixed"
    assert classify("patched the config loader") == "fixed"


def test_classify_decided():
    assert classify("we decided to use sqlite") == "decided"
    assert classify("will use pytest for tests") == "decided"


def test_classify_discovered():
    assert classify("discovered that FastAPI needs async handlers") == "discovered"
    assert classify("turns out OAuth2 requires HTTPS") == "discovered"


def test_classify_error():
    assert classify("Traceback from the last run") == "error"
    assert classify("the build failed on CI") == "error"


def test_classify_no_match():
    assert classify("just a regular sentence") is None
    assert classify("hello world") is None


# ─── Extraction ─────────────────────────────────────────────────────


def test_extract_empty_messages_returns_empty():
    assert extract_observations([]) == []


def test_extract_skips_non_prose_roles():
    msgs = [
        {"role": "system", "content": "I fixed it"},
        {"role": "tool", "content": "I fixed it"},
    ]
    assert extract_observations(msgs) == []


def test_extract_picks_up_salient_lines():
    msgs = [
        {"role": "user", "content": "Can you fix the CORS error?"},
        {"role": "assistant",
         "content": "I fixed the middleware ordering. Discovered that handlers run first."},
    ]
    obs = extract_observations(msgs)
    kinds = [o.kind for o in obs]
    assert "fixed" in kinds
    assert "discovered" in kinds


def test_extract_caps_per_message():
    """More than _MAX_OBS_PER_MESSAGE salient lines are truncated."""
    content = " ".join(f"I fixed bug {i}." for i in range(10))
    obs = extract_observations([{"role": "assistant", "content": content}])
    # Only 2 per message by default.
    assert len(obs) <= 2


def test_extract_includes_summary_as_decided():
    msgs = [{"role": "user", "content": "hello"}]
    obs = extract_observations(msgs, include_summary_text="Refactored the auth flow")
    assert obs[0].kind == "decided"
    assert "Refactored" in obs[0].text


def test_extract_ignores_non_string_content():
    obs = extract_observations([{"role": "assistant", "content": None}])
    assert obs == []


def test_extract_enforces_global_cap():
    """Lots of messages don't produce an unbounded observation list."""
    msgs = [
        {"role": "assistant", "content": f"I fixed bug {i}"}
        for i in range(50)
    ]
    obs = extract_observations(msgs)
    # _MAX_OBS_TOTAL is 12 in flush.py.
    assert len(obs) <= 12


# ─── flush_to_episodic ──────────────────────────────────────────────


def test_flush_writes_observations(mem: MemoryManager):
    msgs = [
        {"role": "assistant", "content": "I fixed the auth bug"},
        {"role": "user", "content": "great"},
    ]
    recorded = flush_to_episodic(
        msgs, mem, session_id="s1", session_title="debug session",
        date="2026-04-07",
    )
    assert len(recorded) >= 1
    text = mem.episodic.read_day("2026-04-07")
    assert "## Session: debug session (s1)" in text
    assert "fixed the auth bug" in text


def test_flush_no_observations_is_noop(mem: MemoryManager):
    msgs = [{"role": "user", "content": "just chatting"}]
    recorded = flush_to_episodic(
        msgs, mem, session_id="s1", session_title="x", date="2026-04-07",
    )
    assert recorded == []
    # No file was created.
    assert mem.episodic.read_day("2026-04-07") == ""


def test_flush_requires_session_id(mem: MemoryManager):
    msgs = [{"role": "assistant", "content": "I fixed it"}]
    recorded = flush_to_episodic(msgs, mem, session_id="")
    assert recorded == []


def test_flush_records_summary_text(mem: MemoryManager):
    recorded = flush_to_episodic(
        [],
        mem,
        session_id="s1",
        session_title="x",
        summary_text="Rewrote the compaction pipeline to be adaptive",
        date="2026-04-07",
    )
    assert len(recorded) == 1
    assert recorded[0].kind == "decided"
    text = mem.episodic.read_day("2026-04-07")
    assert "compaction pipeline" in text


# ─── compact_with_flush ─────────────────────────────────────────────


def test_compact_with_flush_runs_compact_and_records(mem: MemoryManager, make_llm):
    # Build a history that forces stage-2 summarization so we get a
    # non-empty summary_text to record.
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": "u" * 300})
        msgs.append({"role": "assistant",
                     "content": f"I fixed issue {i} and patched the config"})
    llm = make_llm(FakeResponse(FakeMessage(content="refactored auth and deployment")))
    new_msgs, report, recorded = compact_with_flush(
        msgs, llm, mem,
        session_id="s1", session_title="big work",
        config=CompactConfig(target_tokens=200, keep_recent=4),
    )
    # Stage 2 ran.
    assert report.messages_summarized > 0
    assert report.summary_text == "refactored auth and deployment"
    # Observations were written.
    assert len(recorded) >= 1
    assert any("flushed" in n for n in report.notes)
    text = mem.episodic.read_day()
    assert "refactored auth and deployment" in text


def test_compact_with_flush_noop_when_nothing_salient(mem: MemoryManager, make_llm):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    llm = make_llm(FakeResponse(FakeMessage(content="not called")))
    new_msgs, report, recorded = compact_with_flush(
        msgs, llm, mem,
        session_id="s1", session_title="x",
        config=CompactConfig(target_tokens=10_000),
    )
    # Under budget → no compaction work, no observations.
    assert report.messages_summarized == 0
    assert recorded == []


def test_compact_with_flush_returns_compacted_messages(mem: MemoryManager, make_llm):
    msgs = [{"role": "user", "content": "hi"}]
    llm = make_llm(FakeResponse(FakeMessage(content="unused")))
    new_msgs, report, recorded = compact_with_flush(
        msgs, llm, mem,
        session_id="s1", session_title="x",
    )
    # Already under budget → pass-through.
    assert new_msgs == msgs


def test_compact_with_flush_noop_does_not_record_live_history(
    mem: MemoryManager, make_llm,
):
    """Regression: under-budget compaction must not flush the current
    working history to episodic. The prior implementation defaulted
    older=list(messages) when len(messages) <= keep_recent, which
    silently wrote *live, still-in-context* messages into the
    long-term store — including ones with classifier keywords."""
    msgs = [
        {"role": "user", "content": "please help"},
        {"role": "assistant", "content": "I fixed the auth bug last turn"},
        {"role": "user", "content": "thanks"},
    ]
    llm = make_llm(FakeResponse(FakeMessage(content="not called")))
    new_msgs, report, recorded = compact_with_flush(
        msgs, llm, mem,
        session_id="s1", session_title="x",
        config=CompactConfig(target_tokens=10_000),
    )
    assert report.tool_messages_pruned == 0
    assert report.messages_summarized == 0
    assert recorded == []
    # Episodic file must not exist — nothing was discarded, so nothing
    # should have been written.
    assert mem.episodic.read_day() == ""
