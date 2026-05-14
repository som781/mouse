"""Structured event log — one append-only JSONL file per session.

The whole pitch of a generic harness is that its runtime decisions
are structural and inspectable: which tools were exposed, why a
dispatch was refused, how long a tool took, whether the user denied.
Printing those decisions to stderr is fine for a human watching live
but terrible for asking "why did the agent do that" after the fact.
This module gives every harness primitive a single, opaque sink to
record typed events against, so post-hoc analysis ("show me every
grounding refusal across the last N sessions") is a one-liner.

The log is intentionally schema-light:

* The caller supplies an ``event_type`` string and free-form keyword
  data. No event registry, no validation.
* The sink adds a wall-clock timestamp and writes one JSON object
  per line.
* When no path is configured (e.g. no session attached, or the user
  disabled logging), the sink becomes a silent no-op. Harness code
  never has to branch on "is logging enabled?" — it just emits.

The module knows nothing about any specific tool, permission tier,
or domain. It records the harness's own decisions in the harness's
own vocabulary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO


@dataclass
class EventLog:
    """Append-only JSONL sink for harness decision events.

    Construct one per active session. If ``path`` is ``None`` the log
    is a silent no-op — useful when no session is attached or logging
    has been disabled by config. The file handle is opened lazily on
    first :meth:`emit` so instantiating the object never touches disk.
    """

    path: Path | None = None
    _fh: IO | None = field(default=None, repr=False, compare=False)
    _opened: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def null(cls) -> "EventLog":
        """A sink that accepts events and drops them."""
        return cls(path=None)

    def _ensure_open(self) -> None:
        if self._opened or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        except OSError:
            # If the filesystem refuses, degrade to a silent sink
            # rather than crash the agent.
            self._fh = None
        self._opened = True

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
        self._opened = True  # already past the "open on demand" phase

    def emit(self, event_type: str, **data: Any) -> None:
        """Append one event. Never raises — logging must never crash
        the agent loop. A silent sink is better than a half-turn."""
        if self.path is None:
            return
        self._ensure_open()
        if self._fh is None:
            return
        record: dict[str, Any] = {"ts": time.time(), "type": event_type}
        record.update(data)
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except Exception:
            # Losing a single event is preferable to crashing the turn.
            pass
