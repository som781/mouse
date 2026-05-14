"""Adaptive within-session compaction.

The sprint-1 ``/compact`` command was a blunt instrument: summarize
everything. That loses fine-grained context the model still needs
(e.g. the specific paths it just touched). Sprint 3 replaces it with
a two-stage adaptive strategy:

1. **Prune tool outputs.** The context-firewall store already has the
   full output on disk; the copies sitting in ``agent.messages`` are
   duplicated tokens. Replace the content of old/large tool messages
   with a ``[pruned — use fetch_tool_output ref=...]`` marker. This
   is near-free: no LLM call, no loss of information (the model can
   still ``fetch_tool_output`` if it needs the detail back).

2. **Summarize older turns.** If we're still over budget after pruning,
   ask the LLM to summarize everything except the last N messages and
   replace the older half with a single ``[previous conversation
   summary]`` user message.

Each stage is a pure function that takes a message list and returns a
new message list plus a small report, so the agent doesn't need any
special state to drive it and tests can drive each stage in isolation.

Token counting uses a char/4 heuristic — not perfect, but correct
within ~10% for English prose and avoids pulling in a tokenizer just
for a threshold check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Rough chars-per-token for English + code. Undercounts Chinese/Japanese;
# the caller cares about "roughly how much context are we using", not a
# billing-grade number.
CHARS_PER_TOKEN = 4


@dataclass
class CompactReport:
    """What a compaction pass changed."""

    original_tokens: int = 0
    final_tokens: int = 0
    tool_messages_pruned: int = 0
    messages_summarized: int = 0
    summary_text: str = ""
    notes: list[str] = field(default_factory=list)


# ─── Token estimation ───────────────────────────────────────────────


def estimate_tokens(messages: list[dict]) -> int:
    """Rough char/4 token estimate for a message list.

    Sums the length of every ``content`` field plus a tiny fixed
    per-message overhead for role/structural tokens.
    """
    total_chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        # tool_calls also cost tokens when sent to the model.
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                args = fn.get("arguments") or ""
                name = fn.get("name") or ""
                total_chars += len(str(args)) + len(str(name))
        total_chars += 8  # role + separators
    return total_chars // CHARS_PER_TOKEN


# ─── Stage 1: prune tool outputs ────────────────────────────────────


def prune_tool_outputs(
    messages: list[dict],
    *,
    keep_last_n_tools: int = 2,
    min_chars: int = 500,
) -> tuple[list[dict], int]:
    """Replace old/large tool messages with a short reference marker.

    We keep the most recent ``keep_last_n_tools`` tool results intact
    (the model is usually still reasoning about them) and only prune
    ones whose content is over ``min_chars`` — small results aren't
    worth the loss of in-line context.

    The returned list is a shallow copy; callers can swap it into
    ``agent.messages`` without worrying about aliasing.

    Returns ``(new_messages, pruned_count)``.
    """
    # Find the indices of every tool message, in order.
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return list(messages), 0

    # Everything except the last N tool messages is eligible to prune.
    eligible = set(tool_indices[:-keep_last_n_tools] if keep_last_n_tools > 0 else tool_indices)

    out: list[dict] = []
    pruned = 0
    for i, msg in enumerate(messages):
        if i not in eligible:
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < min_chars:
            out.append(msg)
            continue
        tc_id = msg.get("tool_call_id", "")
        omitted = len(content)
        marker = (
            f"[pruned {omitted:,} chars for context savings — "
            f"call fetch_tool_output(tool_call_id={tc_id!r}) to retrieve]"
        )
        out.append({**msg, "content": marker})
        pruned += 1
    return out, pruned


# ─── Tool-call / tool-result pairing invariant ──────────────────────
#
# OpenAI (and most modern providers) reject a message list that
# contains a ``tool`` role whose ``tool_call_id`` was not emitted by a
# preceding ``assistant`` message's ``tool_calls``. They also reject
# an ``assistant`` message whose ``tool_calls`` entries don't all have
# matching ``tool`` responses before the conversation continues.
#
# Stage 2 compaction violates both rules if the keep-recent slice
# happens to land in the middle of a tool_call/tool-result pair: the
# assistant that made the call is summarised away while its response
# still lives in the recent slice (orphan tool response), or vice
# versa (orphan tool_call).
#
# The sanitizer below is a single pass that enforces both invariants
# after any slicing operation. Apply it to any message list that was
# produced by dropping an arbitrary prefix of the history.


def sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
    """Drop orphaned tool responses and unmatched tool_calls.

    Rules enforced:
      * A ``tool`` message is kept only if some earlier ``assistant``
        message in the same list emitted a ``tool_calls`` entry whose
        ``id`` matches its ``tool_call_id``.
      * An ``assistant`` message's ``tool_calls`` list is filtered to
        only those ids that have a matching ``tool`` response later in
        the list. If every call is dropped and the assistant has no
        ``content``, the message itself is removed; otherwise the
        stripped assistant is kept (its prose may still be load-bearing).

    The returned list is a new list; input is not mutated.
    """
    if not messages:
        return list(messages)

    # Pass 1: drop orphan tool messages (call id never emitted earlier).
    emitted_ids: set[str] = set()
    kept: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    if tc_id:
                        emitted_ids.add(tc_id)
            kept.append(m)
        elif role == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id and tc_id in emitted_ids:
                kept.append(m)
            # else: orphan — drop silently
        else:
            kept.append(m)

    # Pass 2: prune tool_calls on assistant messages whose responses
    # didn't survive. Walk the result of pass 1 so we see the true set
    # of tool responses present.
    satisfied_ids: set[str] = {
        m["tool_call_id"] for m in kept
        if m.get("role") == "tool" and m.get("tool_call_id")
    }

    cleaned: list[dict] = []
    for m in kept:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            cleaned.append(m)
            continue
        kept_calls = [
            tc for tc in m["tool_calls"]
            if isinstance(tc, dict) and tc.get("id") in satisfied_ids
        ]
        if kept_calls:
            cleaned.append({**m, "tool_calls": kept_calls})
        elif (m.get("content") or "").strip():
            # Retain the assistant's prose; strip the orphan tool_calls
            new_m = {k: v for k, v in m.items() if k != "tool_calls"}
            cleaned.append(new_m)
        # else: pure orphan assistant (content empty, all calls dropped)
    return cleaned


# ─── Stage 2: summarize older turns ─────────────────────────────────


_SUMMARY_SYSTEM = (
    "You summarize developer assistant conversations compactly and "
    "factually. Include: what the user asked for, what actions were "
    "taken, which files/paths were touched, and the current state. "
    "Omit pleasantries. Use plain prose, ~6-10 sentences."
)


def summarize_older_turns(
    messages: list[dict],
    llm: Any,
    *,
    keep_recent: int = 10,
) -> tuple[list[dict], CompactReport]:
    """Replace all but the most recent ``keep_recent`` messages with a
    single ``[previous summary]`` turn.

    Only the user/assistant/tool content is sent to the LLM for
    summarization — the summary call itself uses a fresh system prompt
    so we never recursively include the one the agent is running with.

    Returns ``(new_messages, report)``. The report's ``summary_text``
    is the model's reply, or empty if the LLM raised.
    """
    report = CompactReport()
    report.original_tokens = estimate_tokens(messages)
    report.final_tokens = report.original_tokens

    # Nothing to do if the history already fits in the recency window.
    if len(messages) <= keep_recent:
        report.notes.append("history shorter than keep_recent; no-op")
        return list(messages), report

    older = messages[:-keep_recent] if keep_recent > 0 else list(messages)
    recent = messages[-keep_recent:] if keep_recent > 0 else []

    # Flatten older turns into a plain-text transcript so the summary
    # call doesn't have to interpret tool_call structure.
    parts: list[str] = []
    for m in older:
        role = m.get("role", "")
        content = m.get("content") or ""
        if not content:
            continue
        parts.append(f"{role.upper()}: {content}")
    transcript = "\n\n".join(parts)[:12000]  # hard cap on summary input

    if not transcript:
        report.notes.append("older messages had no content; dropped without summary")
        report.final_tokens = estimate_tokens(recent)
        report.messages_summarized = len(older)
        return recent, report

    try:
        response = llm.complete(
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",
                 "content": f"Summarize this conversation:\n\n{transcript}"},
            ],
            tools=None,
            tool_choice=None,
            max_tokens=600,
        )
        summary = response.choices[0].message.content or ""
    except Exception as e:
        report.notes.append(f"summarization failed: {e}")
        # Give up on stage 2 rather than destroy history.
        return list(messages), report

    summary = summary.strip()
    if not summary:
        report.notes.append("LLM returned an empty summary; leaving history intact")
        return list(messages), report

    # The ``recent`` slice may start with a ``tool`` message whose
    # matching ``assistant(tool_calls)`` was summarised into ``older``,
    # or it may contain an ``assistant(tool_calls)`` whose tool
    # responses didn't all survive. Either shape is rejected by the
    # provider API. Run the pairing sanitizer before emitting.
    recent_safe = sanitize_tool_pairs(recent)
    dropped_recent = len(recent) - len(recent_safe)
    if dropped_recent:
        report.notes.append(
            f"sanitizer dropped {dropped_recent} orphaned tool pair(s) "
            "from the recent slice"
        )

    compacted: list[dict] = [
        {"role": "user", "content": "[previous conversation summary]"},
        {"role": "assistant", "content": summary},
    ] + recent_safe

    report.summary_text = summary
    report.messages_summarized = len(older)
    report.final_tokens = estimate_tokens(compacted)
    return compacted, report


# ─── Orchestrator ───────────────────────────────────────────────────


@dataclass
class CompactConfig:
    """Knobs for :func:`compact`."""
    target_tokens: int = 6000
    keep_recent: int = 10
    keep_last_n_tools: int = 2
    tool_min_chars: int = 500


def compact(
    messages: list[dict],
    llm: Any,
    config: CompactConfig | None = None,
) -> tuple[list[dict], CompactReport]:
    """Run the adaptive compaction pipeline.

    Stage 1 (prune tool outputs) always runs — it's cheap and
    reference-preserving. Stage 2 (summarize older turns) only runs if
    we're still over the target budget after stage 1.
    """
    cfg = config or CompactConfig()
    report = CompactReport()
    report.original_tokens = estimate_tokens(messages)

    # Stage 1: prune tool outputs. This only rewrites content, so it
    # cannot create pairing orphans — the message list shape is stable.
    pruned_msgs, pruned_count = prune_tool_outputs(
        messages,
        keep_last_n_tools=cfg.keep_last_n_tools,
        min_chars=cfg.tool_min_chars,
    )
    report.tool_messages_pruned = pruned_count
    report.final_tokens = estimate_tokens(pruned_msgs)

    if report.final_tokens <= cfg.target_tokens:
        report.notes.append(
            f"stage 1 sufficient ({report.final_tokens} ≤ {cfg.target_tokens} tokens)"
        )
        return pruned_msgs, report

    # Stage 2: summarize older turns. This drops messages, so the
    # returned list is already run through ``sanitize_tool_pairs`` by
    # ``summarize_older_turns`` to preserve the tool-call/response
    # pairing invariant the provider API enforces.
    summarized_msgs, stage2 = summarize_older_turns(
        pruned_msgs, llm, keep_recent=cfg.keep_recent,
    )
    report.messages_summarized = stage2.messages_summarized
    report.summary_text = stage2.summary_text
    report.final_tokens = stage2.final_tokens
    report.notes.extend(stage2.notes)
    return summarized_msgs, report
