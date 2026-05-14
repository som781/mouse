"""Memory system — episodic, semantic, preferences, and working memory.

Four layers, each with distinct lifetimes (see roadmap §3d):

* **Working memory** — in-session dict, cleared on exit. (Sprint 1 already
  has this as ``Agent.messages``; no module here.)
* **Episodic memory** — cross-session observations in
  ``memory/episodic/YYYY-MM-DD.md``. What happened, what was decided.
* **Preferences** — learned user style/tool preferences in
  ``memory/preferences.json``.
* **Semantic wiki** — Karpathy-style knowledge base in
  ``memory/semantic/*.md``. Facts about the project, ingested on demand.

Each layer is its own module so tests can exercise them in isolation.
``manager.py`` wires them together for the agent.
"""

from mouse.memory.episodic import (
    EpisodicMemory,
    Observation,
    default_episodic_dir,
)
from mouse.memory.preferences import (
    Preferences,
    PreferencesError,
    default_preferences_path,
)
from mouse.memory.semantic import (
    Hit,
    SemanticMemory,
    SemanticMemoryError,
    default_semantic_dir,
    slugify,
)
from mouse.memory.manager import MemoryManager, default_memory_dir
from mouse.memory.injector import InjectionContext, build_memory_context

__all__ = [
    "MemoryManager",
    "default_memory_dir",
    "InjectionContext",
    "build_memory_context",
    "EpisodicMemory",
    "Observation",
    "default_episodic_dir",
    "Preferences",
    "PreferencesError",
    "default_preferences_path",
    "SemanticMemory",
    "SemanticMemoryError",
    "Hit",
    "default_semantic_dir",
    "slugify",
]
