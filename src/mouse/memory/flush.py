"""Pre-compaction memory flush.

When the agent's in-session history is about to be compacted (tool
outputs pruned, older turns summarized), we lose information that
might matter in *future* sessions: what was decided, what was tried,
what went wrong. This module writes a distilled version of that
history into :class:`EpisodicMemory` *before* compaction discards it.

The design is simple enough that tests can drive it without a live
LLM:

* :func:`extract_observations` walks a list of chat messages and
  returns a list of :class:`Observation` objects, classified by
  kind (``decided``, ``fixed``, ``discovered``, ``error``, ``note``).
* :func:`flush_to_episodic` writes those observations into the
  manager's episodic store, tagged with the current session.
* :func:`compact_with_flush` is a convenience wrapper: it calls
  :func:`flush_to_episodic` on the soon-to-be-discarded older
  messages, then delegates to :func:`compact` for the actual
  message-list rewriting.

Keyword-based classification is deliberately dumb. We don't want the
hook to be a second LLM call — that would turn every compaction into
an expensive round-trip. Keywords catch 80% of the "this was
important" signal and the semantic wiki picks up the rest when the
user explicitly ingests.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from mouse.memory.episodic import Observation
from mouse.memory.manager import MemoryManager
from mouse.sessions.compaction import (
    CompactConfig,
    CompactReport,
    compact,
)


# ─── Keyword classifiers ────────────────────────────────────────────


# Leading or mid-sentence markers that imply a given observation kind.
# Order matters — the first match wins, so stronger signals come first.
_CLASSIFIERS: list[tuple[str, re.Pattern[str]]] = [
    ("fixed",      re.compile(r"\b(fixed|resolved|patched|corrected)\b", re.I)),
    ("decided",    re.compile(r"\b(decided|chose to|will use|agreed on)\b", re.I)),
    ("discovered", re.compile(r"\b(discovered|found that|turns out|learned)\b", re.I)),
    ("error",      re.compile(r"\b(error|exception|failed|traceback|stacktrace)\b", re.I)),
]


_MAX_OBS_PER_MESSAGE = 2
_MAX_OBS_TOTAL = 12
_MAX_OBS_LENGTH = 240


def classify(line: str) -> str | None:
    """Return the observation kind for ``line``, or None if nothing matches."""
    for kind, pattern in _CLASSIFIERS:
        if pattern.search(line):
            return kind
    return None


def _pick_salient_lines(content: str) -> list[tuple[str, str]]:
    """Return ``(kind, line)`` for lines in ``content`` that look notable.

    We split on sentences *and* newlines so multi-sentence prose still
    yields one hit per clause. Empty / whitespace lines are skipped.
    """
    if not content or not content.strip():
        return []
    # Split on newlines first, then on sentence boundaries.
    pieces: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces.extend(re.split(r"(?<=[.!?])\s+", line))
    hits: list[tuple[str, str]] = []
    for piece in pieces:
        piece = piece.strip(" -•*\t")
        if not piece:
            continue
        kind = classify(piece)
        if not kind:
            continue
        hits.append((kind, piece[:_MAX_OBS_LENGTH]))
        if len(hits) >= _MAX_OBS_PER_MESSAGE:
            break
    return hits


# ─── Extraction ─────────────────────────────────────────────────────


def extract_observations(
    messages: Iterable[dict],
    *,
    include_summary_text: str = "",
) -> list[Observation]:
    """Walk a message list and build a salient observation list.

    ``include_summary_text`` is the stage-2 compaction summary (if
    any). It's kept as a single high-level ``decided`` observation at
    the top of the list so readers see the distilled story first.
    """
    out: list[Observation] = []

    summary = (include_summary_text or "").strip()
    if summary:
        out.append(Observation(summary[:_MAX_OBS_LENGTH * 2], kind="decided"))

    for msg in messages:
        role = msg.get("role", "")
        # Skip system / tool envelopes — we want actual user/assistant prose.
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for kind, line in _pick_salient_lines(content):
            out.append(Observation(line, kind=kind))
            if len(out) >= _MAX_OBS_TOTAL:
                return out
    return out


# ─── Flush to episodic ──────────────────────────────────────────────


def flush_to_episodic(
    messages: Iterable[dict],
    memory: MemoryManager,
    *,
    session_id: str,
    session_title: str = "",
    summary_text: str = "",
    date: str | None = None,
) -> list[Observation]:
    """Extract observations from ``messages`` and record them.

    Returns the list of observations that were written. Empty if
    nothing in the messages looked worth remembering — in that case
    the episodic store is not touched at all.
    """
    if not session_id:
        # No active session → nowhere to attach the observations.
        return []
    obs = extract_observations(messages, include_summary_text=summary_text)
    if not obs:
        return []
    memory.episodic.record(
        obs,
        session_id=session_id,
        session_title=session_title,
        date=date,
    )
    return obs


# ─── Compact + flush wrapper ────────────────────────────────────────


def compact_with_flush(
    messages: list[dict],
    llm: Any,
    memory: MemoryManager,
    *,
    session_id: str,
    session_title: str = "",
    config: CompactConfig | None = None,
) -> tuple[list[dict], CompactReport, list[Observation]]:
    """Run compaction and flush the discarded-history signal to episodic.

    The older messages we're about to lose are the ones most worth
    remembering. We compute them as ``messages[:-keep_recent]`` — the
    same slice the summarizer uses — and extract observations from
    them before the summarizer overwrites them.

    Returns ``(compacted_messages, report, recorded_observations)``.
    """
    cfg = config or CompactConfig()
    keep = cfg.keep_recent

    # Run the compaction first so we know whether anything actually
    # got discarded. The summary_text (if any) is the single best
    # observation we can record.
    new_messages, report = compact(messages, llm, config=cfg)

    # Only flush observations for history that was actually removed
    # or rewritten. If compaction was a no-op (short history under
    # budget, nothing pruned), there's nothing being lost — recording
    # from the live message list would pollute episodic memory with
    # still-active context.
    nothing_discarded = (
        report.tool_messages_pruned == 0
        and report.messages_summarized == 0
    )
    if nothing_discarded or keep <= 0 or len(messages) <= keep:
        return new_messages, report, []

    older = messages[:-keep]
    recorded = flush_to_episodic(
        older,
        memory,
        session_id=session_id,
        session_title=session_title,
        summary_text=report.summary_text,
    )
    if recorded:
        report.notes.append(f"flushed {len(recorded)} observations to episodic")
    return new_messages, report, recorded
