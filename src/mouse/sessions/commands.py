"""Slash command handlers for session management.

Pulled out of ``cli.py`` so they can be exercised by unit tests without
spinning up the interactive prompt. Each handler takes the objects it
needs (the :class:`SessionManager`, optionally the :class:`Agent`) and
returns a plain string for the caller to print. This keeps the
handlers side-effect-local and makes assertions trivial:

    out = handle_new(mgr, agent, title="debug cors")
    assert "created" in out

``handle_slash_session`` is the dispatcher the CLI calls; it recognises
``/new``, ``/resume``, ``/sessions``, ``/search`` and routes to the
per-command helpers. Returns ``None`` if the command is not one of the
session commands so the CLI can fall through to the next handler.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

from mouse.sessions.manager import SessionManager, SessionManagerError


# Max messages to replay into agent.messages when resuming. We cap to
# keep the context window sane — older turns stay on disk and can be
# surfaced via /search or a future memory layer.
RESUME_MESSAGE_LIMIT = 40


def _fmt_age(ts: float) -> str:
    """Human-friendly relative age for the session list."""
    now = time.time()
    delta = max(0.0, now - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    days = int(delta // 86400)
    if days < 7:
        return f"{days}d ago"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


# ─── Individual handlers ────────────────────────────────────────────


def handle_new(
    mgr: SessionManager,
    agent: Any,
    *,
    title: str = "",
    project_root: str = "",
    model: str = "",
) -> str:
    """Create a new session, reset the agent's in-memory state."""
    rec = mgr.create(
        title=title,
        project_root=project_root,
        model=model or (getattr(agent, "llm", None) and agent.llm.model) or "",
    )
    if agent is not None:
        agent.reset()
        # Sync the session pointer so memory flushes (and any other
        # session-aware machinery) attribute work to the new session.
        if hasattr(agent, "session_id"):
            agent.session_id = rec.id
            agent.session_title = rec.title or ""
    label = rec.title or "(untitled)"
    return f"✓ Created session {rec.id} — {label}"


def handle_resume(
    mgr: SessionManager,
    agent: Any,
    session_id: str,
    *,
    replay_into_agent: bool = True,
) -> str:
    """Resume an existing session, replaying recent messages into the
    agent so the model has context to continue.
    """
    try:
        rec = mgr.resume(session_id)
    except SessionManagerError as e:
        return f"✗ {e}"

    if agent is not None and replay_into_agent:
        agent.reset()
        if hasattr(agent, "session_id"):
            agent.session_id = rec.id
            agent.session_title = rec.title or ""
        msgs = mgr.store.get_messages(rec.id)
        # Only user/assistant roles replay — tool messages reference
        # tool_call_ids that no longer match the reset assistant history,
        # so including them would poison the next turn.
        replayable = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs
            if m["role"] in ("user", "assistant")
        ][-RESUME_MESSAGE_LIMIT:]
        # Route through replace_messages so grounding picks up the
        # replayed content as its substring haystack — otherwise the
        # first follow-up turn would refuse any id the user already
        # discussed in the prior session.
        if hasattr(agent, "replace_messages"):
            agent.replace_messages(replayable)
        else:  # legacy Agent without replace_messages — fall back
            agent.messages = replayable

    label = rec.title or "(untitled)"
    return f"✓ Resumed session {rec.id} — {label}"


def handle_sessions(
    mgr: SessionManager,
    *,
    limit: int = 10,
    formatter: Callable[[Any], str] | None = None,
) -> str:
    """List the most recent sessions, formatted for CLI display."""
    records = mgr.list_sessions(limit=limit)
    if not records:
        return "No sessions yet. Type /new to start one."

    lines = [f"Sessions ({len(records)}):"]
    active_id = mgr.active.id if mgr.active else None
    for rec in records:
        marker = "→ " if rec.id == active_id else "  "
        title = (rec.title or "(untitled)")[:40]
        age = _fmt_age(rec.updated_at)
        lines.append(
            f"{marker}{rec.id}  {title:<40}  {rec.status:<9}  {age}"
        )
    return "\n".join(lines)


def handle_search(mgr: SessionManager, query: str, *, limit: int = 10) -> str:
    """Full-text search across session titles/summaries/tags."""
    query = (query or "").strip()
    if not query:
        return "Usage: /search <query>"
    hits = mgr.search(query, limit=limit)
    if not hits:
        return f"No sessions match {query!r}."
    lines = [f"Matches for {query!r} ({len(hits)}):"]
    for rec in hits:
        title = (rec.title or "(untitled)")[:40]
        age = _fmt_age(rec.updated_at)
        lines.append(f"  {rec.id}  {title:<40}  {age}")
    return "\n".join(lines)


# ─── Dispatcher ──────────────────────────────────────────────────────


def handle_slash_session(
    command: str,
    mgr: SessionManager,
    agent: Any,
) -> str | None:
    """Dispatch ``/new``, ``/resume``, ``/sessions``, ``/search``.

    Returns the message to print, or ``None`` when the command is not
    one of the session commands (so the caller can try other handlers).
    """
    parts = command.strip().split(maxsplit=1)
    if not parts:
        return None
    head = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if head == "/new":
        return handle_new(mgr, agent, title=rest.strip())
    if head == "/resume":
        sid = rest.strip()
        if not sid:
            return "Usage: /resume <session_id>"
        return handle_resume(mgr, agent, sid)
    if head == "/sessions":
        return handle_sessions(mgr)
    if head == "/search":
        return handle_search(mgr, rest)
    return None
