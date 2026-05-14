"""Episodic memory — cross-session observations on disk.

An episodic *observation* is a one-line note the agent writes about
something it learned or did: a discovery, a decision, an error it hit,
a fix that worked. Observations are grouped by date file, then by
session within that file::

    memory/episodic/2026-04-07.md

        # 2026-04-07

        ## Session: Fix auth CORS issue (a1b2c3)
        - Discovered: CORS middleware must be added BEFORE handlers
        - Fixed: Added CORSMiddleware with allow_origins=["*"]

        ## Session: Globex MCP integration (d4e5f6)
        - Connected to Globex MCP server (26 tools discovered)

The file is append-only from the agent's perspective — we never
rewrite earlier sections. New observations for an existing session get
appended under the session's existing header; new sessions get a new
header. This keeps diffs sensible if a user puts the memory dir under
git (which the roadmap encourages).

The module is deliberately dumb: it doesn't try to parse or query the
file. :class:`EpisodicMemory.recent` slurps recent day files and
returns raw markdown blocks; the memory *injector* in Sprint 4 Step 4
is what decides what's relevant for the next turn.

Design notes:
- ``Observation`` carries an explicit ``kind`` tag (``discovered``,
  ``decided``, ``fixed``, ``note``) so renderers and future queries
  can filter. The tags are free-form strings — we don't enforce an
  enum so callers can add categories without touching this module.
- Day files are not file-locked; the agent owns the memory dir and
  concurrent writers are out of scope.
- Dates are UTC to match the transcript filenames (Sprint 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mouse.errors import MouseError


class EpisodicMemoryError(MouseError):
    """Raised on unexpected episodic-memory I/O failures."""


def default_episodic_dir() -> Path:
    """``~/.mouse/memory/episodic`` — the default storage directory."""
    return Path.home() / ".mouse" / "memory" / "episodic"


@dataclass
class Observation:
    """One thing the agent wants to remember across sessions."""

    text: str
    kind: str = "note"  # discovered | decided | fixed | note | ...

    def render(self) -> str:
        """Format as a markdown bullet under a session header."""
        kind = self.kind.strip().capitalize() if self.kind else "Note"
        return f"- {kind}: {self.text.strip()}"


@dataclass
class DayFile:
    """In-memory view of a single ``YYYY-MM-DD.md`` episodic file."""

    date: str  # ISO date, e.g. "2026-04-07"
    path: Path
    raw: str = ""

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def text(self) -> str:
        """Current on-disk contents (empty if not yet created)."""
        if not self.exists:
            return ""
        return self.path.read_text(encoding="utf-8")

    def sessions(self) -> list[str]:
        """Return the list of session ids already present in the file.

        Parses ``## Session: <title> (<id>)`` headers. Used so new
        observations for a known session get appended to the existing
        header instead of creating a duplicate one.
        """
        content = self.text()
        return _parse_session_ids(content)


# ─── Header parsing ──────────────────────────────────────────────────


_SESSION_HEADER_RE = re.compile(
    r"^##\s+Session:\s+(?P<title>.+?)\s+\((?P<sid>[A-Za-z0-9_-]+)\)\s*$",
    re.MULTILINE,
)


def _parse_session_ids(content: str) -> list[str]:
    return [m.group("sid") for m in _SESSION_HEADER_RE.finditer(content)]


def _session_header_bounds(content: str, session_id: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` char offsets of a session's block, or None.

    ``start`` is the beginning of the ``## Session:`` line; ``end`` is
    the end of the block (either the start of the next ``##`` header
    or end-of-file).
    """
    for match in _SESSION_HEADER_RE.finditer(content):
        if match.group("sid") != session_id:
            continue
        start = match.start()
        # Find the next ## header after this one.
        next_header = _SESSION_HEADER_RE.search(content, match.end())
        end = next_header.start() if next_header else len(content)
        return start, end
    return None


# ─── EpisodicMemory ──────────────────────────────────────────────────


class EpisodicMemory:
    """File-backed episodic memory store.

    One instance per agent/CLI process. All writes go through
    :meth:`record`, which handles day-file creation, session-header
    placement, and observation formatting in a single place.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = (
            Path(base_dir).expanduser() if base_dir else default_episodic_dir()
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Day file helpers ──

    @staticmethod
    def today() -> str:
        """Current UTC date as ``YYYY-MM-DD``."""
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    def day_file(self, date: str | None = None) -> DayFile:
        """Return a :class:`DayFile` for ``date`` (default: today)."""
        d = date or self.today()
        return DayFile(date=d, path=self.base_dir / f"{d}.md")

    def list_days(self) -> list[str]:
        """List all dates with observations, oldest-first."""
        if not self.base_dir.exists():
            return []
        dates = []
        for p in sorted(self.base_dir.iterdir()):
            if p.is_file() and p.suffix == ".md":
                stem = p.stem
                # Only count well-formed YYYY-MM-DD names.
                if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
                    dates.append(stem)
        return dates

    # ── Writing ──

    def record(
        self,
        observations: Observation | Iterable[Observation],
        *,
        session_id: str,
        session_title: str = "",
        date: str | None = None,
    ) -> Path:
        """Append one or more observations under a session header.

        If the day file doesn't exist it's created with a top-level
        ``# YYYY-MM-DD`` heading. If the session already has a block
        in the file, new bullets are inserted at the end of that
        block; otherwise a new ``## Session`` header is appended.

        Returns the path of the day file that was written to.
        """
        if isinstance(observations, Observation):
            obs_list = [observations]
        else:
            obs_list = [o for o in observations if isinstance(o, Observation)]
        if not obs_list:
            return self.day_file(date).path

        title = (session_title or "(untitled)").strip()
        if not session_id:
            raise EpisodicMemoryError("record: session_id is required")

        df = self.day_file(date)
        try:
            content = df.text()
            new_content = _append_observations(
                content, df.date, session_id, title, obs_list,
            )
            df.path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            raise EpisodicMemoryError(f"failed to write {df.path}: {e}") from e
        return df.path

    # ── Reading ──

    def read_day(self, date: str | None = None) -> str:
        """Raw markdown for ``date`` (default: today). Empty if missing."""
        return self.day_file(date).text()

    def recent(self, *, days: int = 3) -> list[tuple[str, str]]:
        """Return up to ``days`` most recent day files as ``(date, text)``.

        Newest first. Files that don't exist are skipped. Used by the
        injector to load "what happened lately" into the system prompt.
        """
        if days <= 0:
            return []
        existing = self.list_days()
        # Most-recent-first, capped.
        recent_dates = list(reversed(existing))[:days]
        out: list[tuple[str, str]] = []
        for d in recent_dates:
            text = self.day_file(d).text()
            if text:
                out.append((d, text))
        return out

    def session_observations(
        self, session_id: str, *, date: str | None = None,
    ) -> str:
        """Return the block of text for ``session_id`` on a given day.

        Empty if the session hasn't been recorded. Primarily used by
        handoff.py and the injector when they want just the current
        session's notes.
        """
        content = self.read_day(date)
        bounds = _session_header_bounds(content, session_id)
        if bounds is None:
            return ""
        start, end = bounds
        return content[start:end].rstrip() + "\n"


# ─── Low-level file assembly ────────────────────────────────────────


def _append_observations(
    content: str,
    date: str,
    session_id: str,
    session_title: str,
    observations: list[Observation],
) -> str:
    """Pure function: return new file contents with observations added.

    Three cases:
      1. File is empty → create the ``# YYYY-MM-DD`` header and the
         first session block.
      2. File exists, session not in it → append a new session block.
      3. File exists, session already in it → insert bullets at the
         end of the existing session block.
    """
    bullets = "\n".join(o.render() for o in observations)

    if not content.strip():
        return (
            f"# {date}\n\n"
            f"## Session: {session_title} ({session_id})\n"
            f"{bullets}\n"
        )

    bounds = _session_header_bounds(content, session_id)
    if bounds is None:
        # Append a new session block at the end.
        if not content.endswith("\n"):
            content += "\n"
        if not content.endswith("\n\n"):
            content += "\n"
        return (
            content
            + f"## Session: {session_title} ({session_id})\n"
            + bullets
            + "\n"
        )

    start, end = bounds
    block = content[start:end]
    # Strip trailing blank lines inside the existing block, then tack
    # the new bullets on before the next header (or EOF).
    block = block.rstrip() + "\n" + bullets + "\n"
    # Ensure a blank line separates blocks when another header follows.
    if end < len(content):
        block += "\n"
    return content[:start] + block + content[end:]
