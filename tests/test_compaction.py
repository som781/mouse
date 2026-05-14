"""Tests for the adaptive compaction pipeline."""

from __future__ import annotations

import pytest

from mouse.sessions.compaction import (
    CHARS_PER_TOKEN,
    CompactConfig,
    compact,
    estimate_tokens,
    prune_tool_outputs,
    sanitize_tool_pairs,
    summarize_older_turns,
)

from tests.conftest import FakeMessage, FakeResponse


# ─── Token estimation ────────────────────────────────────────────────


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_estimate_tokens_counts_content_and_overhead():
    msgs = [
        {"role": "user", "content": "a" * 40},
        {"role": "assistant", "content": "b" * 40},
    ]
    # 80 chars of content + 2*8 overhead = 96, / 4 = 24.
    assert estimate_tokens(msgs) == 96 // CHARS_PER_TOKEN


def test_estimate_tokens_counts_tool_call_arguments():
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "t1", "function": {"name": "bash",
                                       "arguments": '{"command": "ls -la"}'}}
        ],
    }]
    # 4 (bash) + 22 (args) + 8 overhead = 34, /4 = 8.
    assert estimate_tokens(msgs) == 8


# ─── Stage 1: prune tool outputs ────────────────────────────────────


def _make_tool_msg(tc_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def test_prune_noop_when_no_tool_messages():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out, pruned = prune_tool_outputs(msgs)
    assert pruned == 0
    assert out == msgs


def test_prune_keeps_last_n_tools_intact():
    big = "x" * 2000
    msgs = [
        _make_tool_msg("t1", big),
        _make_tool_msg("t2", big),
        _make_tool_msg("t3", big),
    ]
    out, pruned = prune_tool_outputs(msgs, keep_last_n_tools=2, min_chars=500)
    # t1 pruned; t2 and t3 kept.
    assert pruned == 1
    assert "pruned" in out[0]["content"]
    assert out[1]["content"] == big
    assert out[2]["content"] == big


def test_prune_skips_small_tool_outputs():
    msgs = [
        _make_tool_msg("t1", "tiny"),
        _make_tool_msg("t2", "tiny"),
        _make_tool_msg("t3", "tiny"),
    ]
    out, pruned = prune_tool_outputs(msgs, keep_last_n_tools=0, min_chars=500)
    # All below min_chars — nothing is pruned.
    assert pruned == 0
    assert out == msgs


def test_prune_marker_includes_tool_call_id():
    msgs = [_make_tool_msg("call_xyz", "y" * 1000)]
    out, pruned = prune_tool_outputs(msgs, keep_last_n_tools=0, min_chars=500)
    assert pruned == 1
    assert "call_xyz" in out[0]["content"]
    assert "fetch_tool_output" in out[0]["content"]


def test_prune_preserves_other_fields():
    msgs = [_make_tool_msg("t1", "z" * 1000)]
    msgs[0]["extra"] = "keepme"
    out, _ = prune_tool_outputs(msgs, keep_last_n_tools=0, min_chars=500)
    assert out[0]["extra"] == "keepme"
    assert out[0]["role"] == "tool"


def test_prune_keep_last_zero_prunes_all_large():
    big = "x" * 1000
    msgs = [_make_tool_msg(f"t{i}", big) for i in range(5)]
    out, pruned = prune_tool_outputs(msgs, keep_last_n_tools=0, min_chars=500)
    assert pruned == 5
    assert all("pruned" in m["content"] for m in out)


# ─── Stage 2: summarize older turns ─────────────────────────────────


def test_summarize_noop_when_history_short(make_llm):
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    llm = make_llm(FakeResponse(FakeMessage(content="SUM")))
    out, report = summarize_older_turns(msgs, llm, keep_recent=10)
    assert out == msgs
    assert report.messages_summarized == 0
    # LLM was never called.
    assert llm.calls == []


def test_summarize_replaces_older_with_synthetic_turn(make_llm):
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    llm = make_llm(FakeResponse(FakeMessage(content="earlier work happened")))
    out, report = summarize_older_turns(msgs, llm, keep_recent=2)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert out[0]["content"] == "[previous conversation summary]"
    assert out[1]["content"] == "earlier work happened"
    assert out[2]["content"] == "u3"
    assert out[3]["content"] == "a3"
    assert report.messages_summarized == 4
    assert "earlier work" in report.summary_text


def test_summarize_handles_llm_failure(make_llm):
    class BrokenLLM:
        def complete(self, **kwargs):
            raise RuntimeError("nope")

    msgs = [{"role": "user", "content": f"u{i}"} for i in range(20)]
    out, report = summarize_older_turns(msgs, BrokenLLM(), keep_recent=5)
    # On failure we return the original history unchanged.
    assert out == msgs
    assert any("failed" in n for n in report.notes)


def test_summarize_empty_reply_leaves_history_intact(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content="")))
    msgs = [{"role": "user", "content": f"u{i}"} for i in range(20)]
    out, report = summarize_older_turns(msgs, llm, keep_recent=5)
    assert out == msgs
    assert any("empty summary" in n for n in report.notes)


def test_summarize_drops_older_when_they_have_no_content(make_llm):
    """When every older message has empty content, there's nothing to
    summarize — we drop them without calling the LLM at all."""
    msgs = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": ""},
        {"role": "user", "content": "recent question"},
    ]
    llm = make_llm(FakeResponse(FakeMessage(content="unused")))
    out, report = summarize_older_turns(msgs, llm, keep_recent=1)
    assert out == [{"role": "user", "content": "recent question"}]
    assert report.messages_summarized == 3
    assert llm.calls == []
    assert any("no content" in n for n in report.notes)


# ─── Orchestrator ───────────────────────────────────────────────────


def test_compact_stage1_sufficient_skips_summary(make_llm):
    """A single big tool output that gets pruned should bring us under
    budget without invoking the summarizer."""
    big = "x" * 40_000  # ~10k tokens
    msgs = [
        {"role": "user", "content": "run ls"},
        {"role": "assistant", "content": ""},
        _make_tool_msg("t1", big),
        {"role": "user", "content": "ok"},
    ]
    llm = make_llm(FakeResponse(FakeMessage(content="SHOULD NOT BE CALLED")))
    out, report = compact(
        msgs, llm,
        config=CompactConfig(target_tokens=500, keep_last_n_tools=0, tool_min_chars=100),
    )
    assert report.tool_messages_pruned == 1
    assert report.messages_summarized == 0
    assert llm.calls == []
    assert any("stage 1 sufficient" in n for n in report.notes)
    assert report.final_tokens < report.original_tokens


def test_compact_stage2_runs_when_still_over_budget(make_llm):
    """If pruning doesn't free enough tokens, the summarizer runs too."""
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": "u" * 300})
        msgs.append({"role": "assistant", "content": "a" * 300})
    llm = make_llm(FakeResponse(FakeMessage(content="earlier back-and-forth")))
    out, report = compact(
        msgs, llm,
        config=CompactConfig(target_tokens=200, keep_recent=4),
    )
    assert report.messages_summarized > 0
    assert report.summary_text == "earlier back-and-forth"
    assert len(out) < len(msgs)
    # The last 4 messages are preserved verbatim.
    assert out[-4:] == msgs[-4:]


def test_compact_already_under_budget_is_noop(make_llm):
    msgs = [{"role": "user", "content": "hi"}]
    llm = make_llm(FakeResponse(FakeMessage(content="should not be called")))
    out, report = compact(msgs, llm, config=CompactConfig(target_tokens=100))
    assert out == msgs
    assert report.tool_messages_pruned == 0
    assert report.messages_summarized == 0
    assert llm.calls == []


# ─── Tool-call / tool-result pairing invariant ──────────────────────
#
# Provider APIs reject a message list that contains:
#   - a ``tool`` message whose ``tool_call_id`` was not emitted by any
#     preceding ``assistant.tool_calls``
#   - an ``assistant.tool_calls`` entry whose matching ``tool`` response
#     is missing
#
# Stage-2 compaction can create either shape when it slices the
# history. The sanitizer below is the guard that keeps a compacted
# history from tripping provider validation.


def test_sanitize_clean_messages_passthrough():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert sanitize_tool_pairs(msgs) == msgs


def test_sanitize_drops_orphan_tool_at_head():
    """The canonical stage-2 failure shape: recent slice starts with a
    tool message whose matching assistant(tool_calls) was summarised."""
    msgs = [
        {"role": "tool", "tool_call_id": "call_old", "content": "orphan"},
        {"role": "user", "content": "continue"},
    ]
    out = sanitize_tool_pairs(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_sanitize_strips_orphan_tool_calls_keeps_content():
    """Assistant with tool_calls that have no matching tool responses
    downstream: strip tool_calls, keep prose."""
    msgs = [
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{"id": "call_a", "function": {"name": "read_file"}}],
        },
        {"role": "user", "content": "continue"},
    ]
    out = sanitize_tool_pairs(msgs)
    assert len(out) == 2
    assert "tool_calls" not in out[0]
    assert out[0]["content"] == "let me check"


def test_sanitize_drops_empty_assistant_with_orphan_tool_calls():
    """Assistant with empty content and only orphan tool_calls is
    wholly unnecessary — drop it."""
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_x", "function": {"name": "bash"}}],
        },
        {"role": "user", "content": "continue"},
    ]
    out = sanitize_tool_pairs(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_sanitize_preserves_valid_pair():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_a", "function": {"name": "read_file"}}],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "file body"},
        {"role": "assistant", "content": "done"},
    ]
    assert sanitize_tool_pairs(msgs) == msgs


def test_sanitize_mixed_orphan_and_valid_pair():
    msgs = [
        {"role": "tool", "tool_call_id": "call_old", "content": "orphan"},
        {"role": "user", "content": "new question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_new", "function": {"name": "bash"}}],
        },
        {"role": "tool", "tool_call_id": "call_new", "content": "new result"},
    ]
    out = sanitize_tool_pairs(msgs)
    assert len(out) == 3
    assert out[0]["role"] == "user"
    assert out[-1]["tool_call_id"] == "call_new"


def test_sanitize_partial_tool_calls_keeps_matched():
    """Assistant emitting two calls where only one has a response:
    strip the orphan, keep the matched call."""
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "function": {"name": "a"}},
                {"id": "call_b", "function": {"name": "b"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "resp a"},
    ]
    out = sanitize_tool_pairs(msgs)
    assert len(out) == 2
    assert [tc["id"] for tc in out[0]["tool_calls"]] == ["call_a"]


def test_summarize_sanitizes_orphan_tool_in_recent(make_llm):
    """Stage 2 end-to-end: slicing creates an orphan at the head of
    ``recent`` and the sanitizer removes it before returning."""
    # 6 older messages (summarised) + 4 recent. The recent slice
    # begins with a tool message whose call was in the older slice.
    older = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "a" * 200},
        {"role": "user", "content": "more"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_dropped", "function": {"name": "x"}}],
        },
        {"role": "user", "content": "keep going"},
        {"role": "assistant", "content": "b" * 200},
    ]
    recent = [
        {"role": "tool", "tool_call_id": "call_dropped", "content": "orphan result"},
        {"role": "user", "content": "last question"},
        {"role": "assistant", "content": "last reply"},
        {"role": "user", "content": "ok thanks"},
    ]
    msgs = older + recent

    llm = make_llm(FakeResponse(FakeMessage(content="summary text")))
    out, report = summarize_older_turns(msgs, llm, keep_recent=4)

    # No tool message in the output should lack a preceding assistant
    # tool_calls with the same id.
    emitted: set[str] = set()
    for m in out:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                emitted.add(tc.get("id", ""))
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in emitted, (
                f"orphan tool message survived sanitizer: {m}"
            )
    assert any("orphan" in n for n in report.notes), report.notes
