"""Memory manager — the façade that owns all three persistent stores.

Four memory layers were defined in ``mouse.memory.__init__``:

1. Working memory — in-session, lives in ``Agent.messages``.
2. Episodic memory — cross-session observations (``episodic.py``).
3. Preferences — learned user style/tool prefs (``preferences.py``).
4. Semantic wiki — hand-curatable knowledge base (``semantic.py``).

This module bundles the three *persistent* layers into a single
:class:`MemoryManager` so the CLI and Agent don't have to wire three
constructors every time. It also centralises the base directory
(``memory/``) so tests can point everything at a tmp path with one
argument.

The manager is deliberately thin — it owns the three stores and a
factory for the default layout; it does not add cross-layer logic.
The context-injection helper that *reads* from all three to build a
prompt preamble lives in :mod:`mouse.memory.injector`.
"""

from __future__ import annotations

from pathlib import Path

from mouse.memory.episodic import EpisodicMemory
from mouse.memory.preferences import Preferences
from mouse.memory.semantic import SemanticMemory


def default_memory_dir() -> Path:
    """``~/.mouse/memory`` — the root of all persistent memory."""
    return Path.home() / ".mouse" / "memory"


class MemoryManager:
    """Owns the episodic, preferences, and semantic stores together.

    Typical use::

        mem = MemoryManager()                # uses ~/.mouse/memory
        mem.episodic.record(obs, session_id=...)
        mem.preferences.set("style.tone", "terse")
        mem.semantic.ingest("auth", "...")

    The constructor eagerly creates the subdirectories so callers
    don't have to worry about race conditions on first write.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = (
            Path(base_dir).expanduser() if base_dir else default_memory_dir()
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.episodic = EpisodicMemory(self.base_dir / "episodic")
        self.preferences = Preferences(self.base_dir / "preferences.json")
        self.semantic = SemanticMemory(self.base_dir / "semantic")

    # ── Convenience ──

    def paths(self) -> dict[str, Path]:
        """Return a mapping of layer name → storage path."""
        return {
            "base": self.base_dir,
            "episodic": self.episodic.base_dir,
            "preferences": self.preferences.path,
            "semantic": self.semantic.base_dir,
        }
