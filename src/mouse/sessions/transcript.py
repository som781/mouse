"""Append-only JSONL transcripts for sessions.

Each session gets one transcript file at
``<base>/transcripts/<date>_<session_id>.jsonl``. Every message written
through :meth:`TranscriptWriter.append` is serialized as a single JSON
object on a single line — never rewritten, never reordered. This is the
source of truth for replay, debugging, and rebuilding the SQLite index
if it's ever lost.

Design notes:
- One file-per-session keeps ``tail -f`` usable during a live session
  and makes cleanup (``rm``) unambiguous.
- We open the file lazily so constructing a writer is cheap, and we
  fsync on ``close()`` only — per-append fsync would cripple throughput
  on SSDs for a dubious durability gain for an interactive tool.
- Each line also carries a monotonically increasing ``seq`` so a reader
  can detect truncation and align with the SQLite index.
- :func:`read_transcript` is a simple whole-file loader used by tests
  and by the replay path in :class:`SessionManager` (Step 4).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from mouse.errors import MouseError


class TranscriptError(MouseError):
    """Raised on transcript read/write failures."""


def transcript_filename(session_id: str, created_at: float | None = None) -> str:
    """Return the canonical filename for a session's transcript.

    Format: ``YYYY-MM-DD_<session_id>.jsonl``. The date is UTC so the
    filename is stable regardless of the reader's timezone.
    """
    ts = created_at if created_at is not None else time.time()
    date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{date}_{session_id}.jsonl"


class TranscriptWriter:
    """Append-only JSONL writer for one session.

    Thread-safety: NOT thread-safe. The agent writes from the main loop
    only. If that ever changes we'll add a lock here instead of pushing
    the concern onto callers.
    """

    def __init__(
        self,
        session_id: str,
        base_dir: str | Path,
        *,
        created_at: float | None = None,
    ) -> None:
        self.session_id = session_id
        self.base_dir = Path(base_dir).expanduser()
        self.transcripts_dir = self.base_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.transcripts_dir / transcript_filename(
            session_id, created_at=created_at
        )
        self._fh = None  # Lazy open on first append.
        self._seq = self._load_last_seq()

    # ── Public API ──

    def append(self, message: dict) -> int:
        """Serialize and append one message. Returns the assigned ``seq``.

        The stored record is::

            {"seq": N, "ts": <unix-float>, ...message fields...}

        We do not validate the message schema here — the transcript
        preserves whatever the agent handed us. Step 4 (SessionManager)
        is the one with a stable contract; this file is the tape.
        """
        if not isinstance(message, dict):
            raise TranscriptError(
                f"append: message must be a dict, got {type(message).__name__}"
            )
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")

        self._seq += 1
        record = {"seq": self._seq, "ts": time.time(), **message}
        try:
            line = json.dumps(record, ensure_ascii=False, default=_json_default)
        except (TypeError, ValueError) as e:
            # Roll back seq so the next append doesn't skip a number.
            self._seq -= 1
            raise TranscriptError(f"append: message is not JSON-serializable: {e}") from e
        self._fh.write(line + "\n")
        self._fh.flush()
        return self._seq

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                # fsync can fail on exotic filesystems (tmpfs, some NFS
                # configs). Flushing is best-effort.
                pass
            self._fh.close()
            self._fh = None

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def seq(self) -> int:
        """The last sequence number written (0 if nothing yet)."""
        return self._seq

    # ── Internals ──

    def _load_last_seq(self) -> int:
        """Scan the existing file (if any) for the highest ``seq`` so
        reopening an existing transcript continues the numbering
        instead of colliding.

        This is O(file size) but only runs once at open. For large
        sessions this could be replaced by a reverse-scan, but that's
        not worth the complexity today.
        """
        if not self.path.exists():
            return 0
        last = 0
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Corrupt line — skip, don't crash the agent.
                        continue
                    s = obj.get("seq")
                    if isinstance(s, int) and s > last:
                        last = s
        except OSError as e:
            raise TranscriptError(f"failed to read existing transcript {self.path}: {e}") from e
        return last


def read_transcript(path: str | Path) -> list[dict]:
    """Load a transcript file into memory.

    Ignores blank lines and silently skips malformed JSON lines — a
    corrupt trailing line (e.g. from a crash mid-write) must not block
    replay of the rest. The caller can diff ``len(lines)`` against the
    highest ``seq`` to detect truncation.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def iter_transcript(path: str | Path) -> Iterator[dict]:
    """Streaming version of :func:`read_transcript` for large files."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _json_default(value: Any) -> Any:
    """Fallback serializer for types that sneak through (e.g. Path)."""
    if isinstance(value, Path):
        return str(value)
    # Let json raise its own TypeError for anything we don't know.
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
