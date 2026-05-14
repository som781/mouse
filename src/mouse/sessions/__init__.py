"""Session persistence: SQLite index + JSONL transcripts.

Sprint 3 adds durable sessions on top of the in-memory agent. The public
entry point is :class:`SessionManager` (added in Step 4); Steps 1–3 build
the storage, transcript, and tool-output primitives it sits on.
"""

from mouse.sessions.storage import SessionStore, SessionRecord
from mouse.sessions.transcript import (
    TranscriptWriter,
    TranscriptError,
    read_transcript,
    iter_transcript,
    transcript_filename,
)
from mouse.sessions.manager import (
    SessionManager,
    SessionManagerError,
    default_sessions_dir,
)

__all__ = [
    "SessionStore",
    "SessionRecord",
    "TranscriptWriter",
    "TranscriptError",
    "read_transcript",
    "iter_transcript",
    "transcript_filename",
    "SessionManager",
    "SessionManagerError",
    "default_sessions_dir",
]
