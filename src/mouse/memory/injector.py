"""Memory injector — build the "what you should remember" preamble.

At session start the agent needs a short block of cross-session context
prepended to its system prompt: recent observations, the user's
standing preferences, and (optionally) a knowledge-base snippet
relevant to whatever the user is about to ask. This module builds that
block from a :class:`MemoryManager`.

The output is plain markdown so the same function can be used for:

* injection into the agent's system prompt,
* a ``/memory`` slash command that prints the current context to stdout,
* logging / debugging what the agent is actually being told.

Design notes:
- Each section is optional. An empty store produces an empty section
  (or is omitted entirely) rather than a placeholder. This keeps the
  preamble small when there's nothing to say — token budgets matter.
- Character budgets are enforced per-section. The overall budget is
  an input so callers (tests, the CLI, a future config option) can
  tune it without editing this file.
- Relevance is deliberately cheap: recent days for episodic, a direct
  substring query for semantic. Embeddings are a future swap-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from mouse.memory.episodic import _session_header_bounds
from mouse.memory.manager import MemoryManager


# ─── Budgets ────────────────────────────────────────────────────────


DEFAULT_EPISODIC_DAYS = 2
DEFAULT_SEMANTIC_HITS = 3
DEFAULT_EPISODIC_CHARS = 2000
DEFAULT_PREFERENCES_CHARS = 800
DEFAULT_SEMANTIC_CHARS = 1500


@dataclass
class InjectionContext:
    """Structured result of :func:`build_memory_context`.

    The text lives in :attr:`markdown`; the individual sections are
    also exposed so callers can assemble them differently (e.g. a UI
    that renders them in separate panels).
    """

    markdown: str
    episodic: str
    preferences: str
    semantic: str

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()


# ─── Section builders ───────────────────────────────────────────────


def _strip_session_block(text: str, session_id: str) -> str:
    """Remove the ``## Session: … (session_id)`` block from ``text``.

    Used by :func:`build_episodic_section` to avoid re-injecting the
    *active* session's own observations — they're already live in
    working memory and echoing them back as "memory context" wastes
    tokens and confuses the model.
    """
    bounds = _session_header_bounds(text, session_id)
    if bounds is None:
        return text
    start, end = bounds
    return (text[:start] + text[end:]).rstrip()


def build_episodic_section(
    mem: MemoryManager,
    *,
    days: int = DEFAULT_EPISODIC_DAYS,
    max_chars: int = DEFAULT_EPISODIC_CHARS,
    exclude_session: str = "",
) -> str:
    """Recent observations across the last ``days`` day files.

    If ``exclude_session`` is provided, that session's block is
    stripped from each day before formatting — callers pass the
    active session's id so the agent doesn't get its own working-
    memory content recycled back to it as "recent observations".
    """
    recent = mem.episodic.recent(days=days)
    if not recent:
        return ""
    chunks: list[str] = []
    used = 0
    for date, text in recent:
        if exclude_session:
            text = _strip_session_block(text, exclude_session)
        text = text.strip()
        if not text:
            continue
        # A day with only its ``# YYYY-MM-DD`` header and no session
        # blocks carries no real content — skip it rather than emit
        # a dangling heading.
        if "## Session:" not in text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        block = text if len(text) <= remaining else text[:remaining] + "…"
        chunks.append(block)
        used += len(block)
    if not chunks:
        return ""
    body = "\n\n".join(chunks)
    return f"## Recent observations\n\n{body}"


def build_preferences_section(
    mem: MemoryManager,
    *,
    max_chars: int = DEFAULT_PREFERENCES_CHARS,
) -> str:
    """The user's persistent preferences as flat ``key: value`` lines."""
    data = mem.preferences.all()
    if not data:
        return ""
    lines: list[str] = []
    for key, value in _flatten(data):
        line = f"- {key}: {_format_value(value)}"
        lines.append(line)
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit("\n", 1)[0] + "\n- …"
    return f"## User preferences\n\n{body}"


def build_semantic_section(
    mem: MemoryManager,
    hint: str,
    *,
    limit: int = DEFAULT_SEMANTIC_HITS,
    max_chars: int = DEFAULT_SEMANTIC_CHARS,
) -> str:
    """Top knowledge-base snippets matching ``hint`` (empty if none)."""
    if not hint or not hint.strip():
        return ""
    hits = mem.semantic.query(hint, limit=limit)
    if not hits:
        return ""
    chunks: list[str] = []
    used = 0
    for h in hits:
        entry = f"**{h.title}** ({h.topic}) — {h.snippet}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(entry) > remaining:
            entry = entry[:remaining] + "…"
        chunks.append(entry)
        used += len(entry) + 2  # rough newline accounting
    if not chunks:
        return ""
    return "## Relevant knowledge\n\n" + "\n\n".join(chunks)


# ─── Top-level builder ──────────────────────────────────────────────


def build_memory_context(
    mem: MemoryManager,
    *,
    hint: str = "",
    days: int = DEFAULT_EPISODIC_DAYS,
    semantic_limit: int = DEFAULT_SEMANTIC_HITS,
    exclude_session: str = "",
) -> InjectionContext:
    """Assemble the full memory preamble.

    ``hint`` steers the semantic query — the CLI passes the user's
    first message (or session title) so knowledge lookups are actually
    relevant to what's about to happen. If empty, the semantic section
    is skipped.

    ``exclude_session`` should be the active session's id, so its own
    observations aren't re-injected as "recent" — they're already in
    working memory.
    """
    episodic = build_episodic_section(mem, days=days, exclude_session=exclude_session)
    preferences = build_preferences_section(mem)
    semantic = build_semantic_section(mem, hint, limit=semantic_limit)

    sections = [s for s in (episodic, preferences, semantic) if s]
    if not sections:
        return InjectionContext("", "", "", "")
    markdown = "# Memory context\n\n" + "\n\n".join(sections)
    return InjectionContext(
        markdown=markdown,
        episodic=episodic,
        preferences=preferences,
        semantic=semantic,
    )


# ─── Helpers ────────────────────────────────────────────────────────


def _flatten(data: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten nested dicts to ``[(dotted.key, value), ...]``."""
    out: list[tuple[str, object]] = []
    for k, v in sorted(data.items()):
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten(v, prefix=key))
        else:
            out.append((key, v))
    return out


def _format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)
    return str(value)
