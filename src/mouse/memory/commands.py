"""Slash command handlers for memory operations.

Separated from ``cli.py`` so they can be unit-tested without a live
prompt. Each handler takes the memory objects it needs and returns a
plain string for the CLI to print.

``handle_slash_knowledge`` is the dispatcher for ``/knowledge``
sub-commands::

    /knowledge ingest <topic> <content...>
    /knowledge query <text...>
    /knowledge list
    /knowledge show <topic>
    /knowledge delete <topic>

Returns ``None`` for anything that isn't a recognised knowledge
command so the CLI can fall through to the next handler.
"""

from __future__ import annotations

from mouse.memory.semantic import SemanticMemory, SemanticMemoryError


_USAGE = (
    "Usage:\n"
    "  /knowledge ingest <topic> <content>\n"
    "  /knowledge query <text>\n"
    "  /knowledge list\n"
    "  /knowledge show <topic>\n"
    "  /knowledge delete <topic>"
)


def handle_ingest(mem: SemanticMemory, rest: str) -> str:
    """Parse ``<topic> <content>`` and write it into the wiki."""
    rest = rest.strip()
    if not rest:
        return "Usage: /knowledge ingest <topic> <content>"
    parts = rest.split(maxsplit=1)
    if len(parts) < 2:
        return "Usage: /knowledge ingest <topic> <content>"
    topic, content = parts[0], parts[1]
    try:
        mem.ingest(topic, content, title=topic)
    except SemanticMemoryError as e:
        return f"✗ {e}"
    return f"✓ Ingested into {topic!r}"


def handle_query(mem: SemanticMemory, rest: str, *, limit: int = 5) -> str:
    """Search the wiki and format the top hits."""
    rest = rest.strip()
    if not rest:
        return "Usage: /knowledge query <text>"
    hits = mem.query(rest, limit=limit)
    if not hits:
        return f"No knowledge matches {rest!r}."
    lines = [f"Matches for {rest!r} ({len(hits)}):"]
    for h in hits:
        lines.append(f"  [{h.score}] {h.topic} — {h.title}")
        lines.append(f"        {h.snippet}")
    return "\n".join(lines)


def handle_list(mem: SemanticMemory) -> str:
    """List all topics in the wiki."""
    topics = mem.list_topics()
    if not topics:
        return "No knowledge topics yet. Use /knowledge ingest to add one."
    lines = [f"Knowledge topics ({len(topics)}):"]
    for slug in topics:
        lines.append(f"  {slug} — {mem.title_of(slug)}")
    return "\n".join(lines)


def _first_token(rest: str) -> str:
    """Return the first whitespace-separated token, or ``""``.

    ``show``/``delete`` take a single topic slug; the dispatcher hands
    us the whole tail after the subcommand, but we only care about
    the first word — extra args would otherwise get slugified into a
    bogus topic like ``auth-flow`` that doesn't match anything.
    """
    parts = rest.strip().split()
    return parts[0] if parts else ""


def handle_show(mem: SemanticMemory, topic: str) -> str:
    """Print the raw markdown for a topic."""
    topic = _first_token(topic)
    if not topic:
        return "Usage: /knowledge show <topic>"
    text = mem.get(topic)
    if not text:
        return f"No topic {topic!r}."
    return text


def handle_delete(mem: SemanticMemory, topic: str) -> str:
    """Remove a topic from the wiki."""
    topic = _first_token(topic)
    if not topic:
        return "Usage: /knowledge delete <topic>"
    try:
        removed = mem.delete(topic)
    except SemanticMemoryError as e:
        return f"✗ {e}"
    if not removed:
        return f"No topic {topic!r}."
    return f"✓ Deleted {topic!r}"


# ─── Dispatcher ──────────────────────────────────────────────────────


def handle_slash_knowledge(
    command: str,
    mem: SemanticMemory,
) -> str | None:
    """Dispatch ``/knowledge <subcommand> …``. Returns ``None`` if the
    command isn't ``/knowledge``."""
    parts = command.strip().split(maxsplit=2)
    if not parts or parts[0].lower() != "/knowledge":
        return None
    if len(parts) == 1:
        return _USAGE
    sub = parts[1].lower()
    rest = parts[2] if len(parts) > 2 else ""

    if sub == "ingest":
        return handle_ingest(mem, rest)
    if sub == "query":
        return handle_query(mem, rest)
    if sub == "list":
        return handle_list(mem)
    if sub == "show":
        return handle_show(mem, rest)
    if sub == "delete":
        return handle_delete(mem, rest)
    return f"Unknown /knowledge subcommand: {sub!r}\n\n{_USAGE}"
