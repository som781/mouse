"""SessionManager — the façade that wires storage, transcript, and tool
output persistence together into one coherent object.

A session's on-disk layout is::

    <base_dir>/
        sessions.db                                # SQLite index (storage.py)
        transcripts/<date>_<id>.jsonl              # Append-only log (transcript.py)
        sessions/<id>/tool_outputs/<tc_id>.txt     # Context-firewall store

``SessionManager`` has two lifecycles:

1. **Process-level** (``__init__`` / ``close``): opens the SQLite store.
   Cheap. Create one at CLI startup, close at exit.
2. **Session-level** (``create`` / ``resume`` / ``end``): switches the
   currently active session — opens a transcript writer, points the
   tool-output store at the session's disk directory, and tracks the
   session id so ``record_*`` calls know where to write.

Only one session is "active" at a time in a given manager. ``create``
while another session is active transparently ends it first.

Auto-titling: after the first user+assistant exchange, ``auto_title``
asks the LLM to produce a short label and persists it to the sessions
row. Calling it again is a no-op if a non-empty title already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mouse.errors import MouseError
from mouse.sessions.storage import SessionRecord, SessionStore
from mouse.sessions.transcript import TranscriptWriter
from mouse.tools import tool_output_store


class SessionManagerError(MouseError):
    """Raised on invalid session manager state transitions."""


# Cap the auto-generated title so it fits in narrow CLI listings.
MAX_TITLE_CHARS = 60


_AUTO_TITLE_SYSTEM = (
    "You generate short, descriptive titles (<= 60 characters) that "
    "summarize the topic of a developer assistant conversation. "
    "Reply with ONLY the title — no quotes, no prefix, no trailing punctuation."
)


@dataclass
class _ActiveSession:
    """Bookkeeping for the currently open session."""

    record: SessionRecord
    writer: TranscriptWriter


def default_sessions_dir() -> Path:
    """``~/.mouse/sessions`` — the default base directory."""
    return Path.home() / ".mouse" / "sessions"


class SessionManager:
    """Create, resume, list, and search sessions.

    The manager owns one open :class:`SessionStore` for its lifetime.
    It may optionally have one active session (created via ``create``
    or ``resume``). All the ``record_*`` helpers write to both the
    SQLite index and the JSONL transcript, so a crash never leaves the
    two out of sync beyond the last in-flight write.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir).expanduser() if base_dir else default_sessions_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.store = SessionStore(self.base_dir / "sessions.db")
        self._active: _ActiveSession | None = None

    # ── Lifecycle ──

    def close(self) -> None:
        """Close the active session (if any) and the store."""
        self.end()
        self.store.close()

    def __enter__(self) -> SessionManager:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def active(self) -> SessionRecord | None:
        return self._active.record if self._active else None

    # ── Session creation / resume / end ──

    def create(
        self,
        *,
        project_root: str = "",
        model: str = "",
        title: str = "",
    ) -> SessionRecord:
        """Start a new session and make it active."""
        if self._active is not None:
            self.end()
        rec = self.store.create_session(
            project_root=project_root, model=model, title=title,
        )
        self._open_active(rec)
        return rec

    def resume(self, session_id: str) -> SessionRecord:
        """Make an existing session active. Raises if it's unknown."""
        if self._active is not None and self._active.record.id == session_id:
            return self._active.record
        if self._active is not None:
            self.end()
        rec = self.store.get_session(session_id)
        if rec is None:
            raise SessionManagerError(f"unknown session id: {session_id!r}")
        self._open_active(rec)
        return rec

    def end(self) -> None:
        """Close the active session (flush transcript, detach tool store)."""
        if self._active is None:
            return
        try:
            self._active.writer.close()
        finally:
            tool_output_store.configure(None)
            self._active = None

    def _open_active(self, rec: SessionRecord) -> None:
        writer = TranscriptWriter(
            rec.id, self.base_dir, created_at=rec.created_at,
        )
        session_dir = self.base_dir / "sessions" / rec.id
        tool_output_store.configure(session_dir)
        self._active = _ActiveSession(record=rec, writer=writer)

    def session_dir(self, session_id: str) -> Path:
        """On-disk directory that co-locates a session's artifacts
        (tool outputs, event log, future per-session caches)."""
        return self.base_dir / "sessions" / session_id

    def events_path(self, session_id: str) -> Path:
        """Path to the append-only JSONL event log for a session.

        Co-located with the tool-output store so a single rm -r per
        session id removes everything the harness wrote for that run.
        """
        return self.session_dir(session_id) / "events.jsonl"

    # ── Query / listing ──

    def list_sessions(self, limit: int = 20) -> list[SessionRecord]:
        return self.store.list_sessions(limit=limit)

    def search(self, query: str, limit: int = 20) -> list[SessionRecord]:
        return self.store.search(query, limit=limit)

    def get(self, session_id: str) -> SessionRecord | None:
        return self.store.get_session(session_id)

    def delete(self, session_id: str) -> None:
        """Delete a session from the index.

        Transcripts and tool-output files on disk are left in place —
        they're an audit trail. A future ``purge`` command can scrub
        them when users explicitly want that.
        """
        if self._active is not None and self._active.record.id == session_id:
            self.end()
        self.store.delete_session(session_id)

    # ── Recording (active session only) ──

    def _require_active(self) -> _ActiveSession:
        if self._active is None:
            raise SessionManagerError("no active session — call create() or resume() first")
        return self._active

    def record_user(self, content: str, *, token_count: int = 0) -> None:
        active = self._require_active()
        self.store.append_message(
            active.record.id, role="user", content=content, token_count=token_count,
        )
        active.writer.append({"role": "user", "content": content, "token_count": token_count})

    def record_assistant(self, content: str, *, token_count: int = 0) -> None:
        active = self._require_active()
        self.store.append_message(
            active.record.id, role="assistant", content=content, token_count=token_count,
        )
        active.writer.append(
            {"role": "assistant", "content": content, "token_count": token_count}
        )

    def record_tool(
        self,
        *,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        tool_call_id: str = "",
    ) -> None:
        active = self._require_active()
        self.store.append_message(
            active.record.id,
            role="tool",
            content=tool_output,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
        )
        active.writer.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
        })

    def update_token_count(self, total: int) -> None:
        active = self._require_active()
        self.store.update_session(active.record.id, total_tokens=total)
        # Keep our in-memory copy roughly current.
        active.record.total_tokens = total

    def set_title(self, title: str) -> None:
        active = self._require_active()
        clean = title.strip()[:MAX_TITLE_CHARS]
        self.store.update_session(active.record.id, title=clean)
        active.record.title = clean

    def set_summary(self, summary: str) -> None:
        active = self._require_active()
        self.store.update_session(active.record.id, summary=summary)
        active.record.summary = summary

    # ── Auto-title ──

    def auto_title(self, llm: Any, *, force: bool = False) -> str:
        """Ask the LLM for a short title for the current session.

        Reads the first user/assistant messages from the SQLite store
        (so it works during resume too), sends a minimal prompt, and
        persists the result. Returns the title that was set, or the
        existing title if one is already present (unless ``force``).
        """
        active = self._require_active()
        if active.record.title and not force:
            return active.record.title

        msgs = self.store.get_messages(active.record.id)
        first_user = next((m for m in msgs if m["role"] == "user"), None)
        first_assistant = next((m for m in msgs if m["role"] == "assistant"), None)
        if first_user is None:
            # Nothing to summarize yet.
            return ""

        prompt_parts = [f"USER: {first_user['content'][:500]}"]
        if first_assistant is not None:
            prompt_parts.append(f"ASSISTANT: {first_assistant['content'][:500]}")
        prompt_parts.append("Title:")
        user_prompt = "\n\n".join(prompt_parts)

        try:
            response = llm.complete(
                messages=[
                    {"role": "system", "content": _AUTO_TITLE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
                tool_choice=None,
                max_tokens=40,
            )
        except Exception:
            # Titling failures are never fatal; fall back to the first
            # few words of the user message so the CLI has something
            # better than a bare session id.
            fallback = first_user["content"].strip().splitlines()[0][:MAX_TITLE_CHARS]
            self.set_title(fallback)
            return fallback

        title = ""
        try:
            title = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            title = ""
        title = title.strip().strip('"').strip("'")
        if not title:
            title = first_user["content"].strip().splitlines()[0]
        self.set_title(title)
        return active.record.title
