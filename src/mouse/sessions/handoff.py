"""Progress.md handoff generator — structured handoff between sessions.

Anthropic's research on long-running agent tasks found that clean
context *resets* between phases beat in-place compaction: the next
session gets a fresh prompt plus a small, structured progress file
instead of a summarized transcript. This module generates that file.

The output format is fixed (per ``openharness-roadmap-v2.md`` §3c):

    # Progress
    ## Completed
    - [x] ...
    ## In Progress
    - [ ] ...
    ## Discovered
    - ...
    ## Files Modified
    - ...

We collect the raw session history from :class:`SessionManager`, hand
it to the LLM with a tightly-scoped prompt, and write the markdown to
``<base_dir>/sessions/<session_id>/progress.md``. Callers get the
written path back so they can show it to the user or open it.

Tests drive the pure functions (``build_handoff_prompt``,
``generate_progress``) with a scripted LLM so no real completion is
ever made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mouse.errors import MouseError
from mouse.sessions.manager import SessionManager


class HandoffError(MouseError):
    """Raised when handoff generation fails in a non-recoverable way."""


# Hard cap on the transcript size fed to the LLM. Past this, we trim
# the oldest messages — the progress file is meant to capture state,
# not replay the entire session.
MAX_TRANSCRIPT_CHARS = 16_000


_PROGRESS_SYSTEM = (
    "You write structured progress.md files for an AI coding assistant. "
    "Read the conversation and produce a Markdown document with exactly "
    "these top-level sections, in this order:\n"
    "  # Progress\n"
    "  ## Completed\n"
    "  ## In Progress\n"
    "  ## Discovered\n"
    "  ## Files Modified\n"
    "Use '- [x]' for completed tasks, '- [ ]' for in-progress tasks, "
    "and '- ' bullets under Discovered and Files Modified. Be specific: "
    "list concrete file paths, decisions, and observations. If a section "
    "has nothing to report, include the heading with an explicit "
    "'_None yet._' line underneath. Output ONLY the markdown — no "
    "code fences, no preamble, no trailing commentary."
)


_FALLBACK_PROGRESS = """# Progress

## Completed
_None yet._

## In Progress
_None yet._

## Discovered
_None yet._

## Files Modified
_None yet._
"""


# ─── Transcript assembly ────────────────────────────────────────────


def _format_message_for_prompt(msg: dict) -> str:
    """Render one stored message as a flat line for the handoff prompt."""
    role = (msg.get("role") or "").upper()
    if role == "TOOL":
        name = msg.get("tool_name") or "tool"
        inp = msg.get("tool_input") or ""
        out = msg.get("tool_output") or msg.get("content") or ""
        # Truncate noisy tool output — the LLM only needs the gist.
        if len(out) > 400:
            out = out[:400] + "...(truncated)"
        return f"TOOL[{name}]({inp}): {out}"
    content = (msg.get("content") or "").strip()
    if not content:
        return ""
    return f"{role}: {content}"


def build_handoff_prompt(messages: list[dict]) -> str:
    """Return the user-message payload for the handoff LLM call.

    The transcript is formatted as plain text (not JSON) because the
    LLM is better at pattern-matching structured prose than nested
    dicts, and we already control the schema of the output.
    """
    rendered = [_format_message_for_prompt(m) for m in messages]
    transcript = "\n\n".join(p for p in rendered if p.strip())
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        # Keep the TAIL — the most recent work is the most relevant to
        # "what's in progress" and "what's discovered".
        transcript = "...(older context trimmed)...\n\n" + transcript[-MAX_TRANSCRIPT_CHARS:]
    return (
        "Generate a progress.md for the following session transcript. "
        "Follow the section structure from the system prompt exactly.\n\n"
        f"{transcript}"
    )


# ─── Generation ─────────────────────────────────────────────────────


def generate_progress(messages: list[dict], llm: Any) -> str:
    """Ask the LLM to produce a progress.md and return its text.

    Falls back to an empty skeleton if the LLM errors or returns blank.
    The skeleton keeps downstream tooling (``save_progress``, editors)
    working instead of failing mid-handoff.
    """
    if not messages:
        return _FALLBACK_PROGRESS

    prompt = build_handoff_prompt(messages)
    try:
        response = llm.complete(
            messages=[
                {"role": "system", "content": _PROGRESS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            tool_choice=None,
            max_tokens=1200,
        )
    except Exception:
        return _FALLBACK_PROGRESS

    try:
        text = response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        text = ""
    text = text.strip()
    if not text:
        return _FALLBACK_PROGRESS
    if not text.lstrip().startswith("# "):
        text = "# Progress\n\n" + text
    return text + ("\n" if not text.endswith("\n") else "")


# ─── Persistence ────────────────────────────────────────────────────


def progress_path(base_dir: Path, session_id: str) -> Path:
    """Return the canonical on-disk path for a session's progress.md."""
    return base_dir / "sessions" / session_id / "progress.md"


def save_progress(mgr: SessionManager, content: str) -> Path:
    """Write a progress.md for the currently active session.

    Raises :class:`HandoffError` if no session is active — callers
    should run this after a ``create`` or ``resume``.
    """
    if mgr.active is None:
        raise HandoffError("no active session — cannot save progress.md")
    path = progress_path(mgr.base_dir, mgr.active.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ─── CLI entry point ────────────────────────────────────────────────


def handle_progress(mgr: SessionManager, agent: Any) -> str:
    """``/progress`` slash-command handler.

    Pulls all messages for the active session from the store, asks the
    agent's LLM to generate a progress.md, saves it next to the
    session's tool outputs, and returns a one-line status string for
    the CLI to print.
    """
    if mgr.active is None:
        return "✗ No active session — type /new to start one."
    messages = mgr.store.get_messages(mgr.active.id)
    if not messages:
        return "No messages yet — nothing to hand off."

    llm = getattr(agent, "llm", None)
    if llm is None:
        return "✗ Agent has no LLM attached."

    content = generate_progress(messages, llm)
    path = save_progress(mgr, content)
    return f"✓ Wrote progress.md → {path}"
