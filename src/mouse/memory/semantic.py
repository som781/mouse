"""Semantic memory — a tiny Karpathy-style markdown knowledge base.

Semantic memory holds *facts the agent knows about the project*: how the
auth flow works, where the config files live, which services talk to
which. Unlike episodic memory (what happened on 2026-04-07), semantic
entries are timeless and hand-curatable::

    memory/semantic/auth.md
    memory/semantic/deployment.md
    memory/semantic/data-pipeline.md

Each file is a markdown note keyed by a *topic slug*. Slugs are derived
from the topic title by lowercasing, stripping non-alphanumerics, and
collapsing whitespace to hyphens. Files are append-friendly: calling
:meth:`SemanticMemory.ingest` on an existing topic appends a new
section under a timestamp separator, so the history of a topic grows
without overwriting prior context.

Querying is deliberately dumb — a substring scan over title + body,
returning ``(topic, snippet)`` hits ranked by match count. This keeps
the module dependency-free; a future step can swap in embeddings
without touching callers if we keep the query signature stable.

Design notes:
- One instance per agent process, rooted at ``memory/semantic/``.
- Topic slugs are the source of truth. The human-readable title is
  stored as an ``# H1`` inside the file for display only.
- No file locking; single-writer assumption matches the rest of the
  memory system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mouse.errors import MouseError


class SemanticMemoryError(MouseError):
    """Raised on semantic-memory I/O or validation failures."""


def default_semantic_dir() -> Path:
    """``~/.mouse/memory/semantic`` — the default storage directory."""
    return Path.home() / ".mouse" / "memory" / "semantic"


@dataclass
class Hit:
    """One search result from :meth:`SemanticMemory.query`."""

    topic: str          # slug, e.g. "auth"
    title: str          # display title from the H1
    snippet: str        # short excerpt around the first match
    score: int          # number of matches (higher = more relevant)


# ─── Slug handling ──────────────────────────────────────────────────


_SLUG_STRIP_RE = re.compile(r"[^a-z0-9\s_-]+")
_SLUG_SPACE_RE = re.compile(r"[\s_-]+")


def slugify(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace to hyphens."""
    if not isinstance(title, str):
        raise SemanticMemoryError("topic title must be a string")
    s = title.strip().lower()
    s = _SLUG_STRIP_RE.sub("", s)
    s = _SLUG_SPACE_RE.sub("-", s).strip("-")
    if not s:
        raise SemanticMemoryError(f"topic title {title!r} produced an empty slug")
    return s


# ─── SemanticMemory ─────────────────────────────────────────────────


class SemanticMemory:
    """Markdown-file-backed semantic knowledge base."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = (
            Path(base_dir).expanduser() if base_dir else default_semantic_dir()
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Paths ──

    def topic_path(self, topic: str) -> Path:
        """Return the ``.md`` path for a slug (does not check existence)."""
        slug = slugify(topic)
        return self.base_dir / f"{slug}.md"

    def list_topics(self) -> list[str]:
        """All topic slugs in the store, sorted alphabetically."""
        if not self.base_dir.exists():
            return []
        return sorted(
            p.stem for p in self.base_dir.iterdir()
            if p.is_file() and p.suffix == ".md"
        )

    def exists(self, topic: str) -> bool:
        return self.topic_path(topic).exists()

    # ── Write ──

    def ingest(
        self,
        topic: str,
        content: str,
        *,
        title: str | None = None,
        timestamp: str | None = None,
    ) -> Path:
        """Append ``content`` to the topic's file (creating it if needed).

        On first write the file gets an ``# {title}`` header. Subsequent
        writes append under a ``---`` separator with a timestamp line.
        Returns the path of the file written.
        """
        if not isinstance(content, str) or not content.strip():
            raise SemanticMemoryError("ingest: content must be a non-empty string")
        path = self.topic_path(topic)
        display_title = (title or topic).strip()
        ts = timestamp or self._now()
        body = content.strip() + "\n"
        try:
            if path.exists():
                prior = path.read_text(encoding="utf-8")
                if not prior.endswith("\n"):
                    prior += "\n"
                new_text = prior + f"\n---\n_ingested {ts}_\n\n" + body
            else:
                new_text = f"# {display_title}\n\n_ingested {ts}_\n\n{body}"
            path.write_text(new_text, encoding="utf-8")
        except OSError as e:
            raise SemanticMemoryError(f"failed to write {path}: {e}") from e
        return path

    def delete(self, topic: str) -> bool:
        """Remove a topic file. Returns True if it existed."""
        path = self.topic_path(topic)
        if not path.exists():
            return False
        try:
            path.unlink()
        except OSError as e:
            raise SemanticMemoryError(f"failed to delete {path}: {e}") from e
        return True

    # ── Read ──

    def get(self, topic: str) -> str:
        """Return the raw markdown for a topic, or empty string if missing."""
        path = self.topic_path(topic)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise SemanticMemoryError(f"failed to read {path}: {e}") from e

    def title_of(self, topic: str) -> str:
        """Display title (H1) for a topic, or the slug if there's no H1."""
        text = self.get(topic)
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return slugify(topic)

    # ── Query ──

    def query(self, needle: str, *, limit: int = 5) -> list[Hit]:
        """Rank topics by case-insensitive substring matches of ``needle``.

        Returns up to ``limit`` hits, highest-score first. Ties break on
        topic slug alphabetically so results are deterministic.
        """
        if not isinstance(needle, str) or not needle.strip():
            return []
        term = needle.strip().lower()
        hits: list[Hit] = []
        for slug in self.list_topics():
            text = self.get(slug)
            if not text:
                continue
            score = text.lower().count(term)
            if score == 0:
                continue
            hits.append(
                Hit(
                    topic=slug,
                    title=self.title_of(slug),
                    snippet=_snippet_around(text, term),
                    score=score,
                )
            )
        hits.sort(key=lambda h: (-h.score, h.topic))
        return hits[:limit]

    # ── Bulk ──

    def ingest_many(self, items: Iterable[tuple[str, str]]) -> list[Path]:
        """Ingest a list of ``(topic, content)`` pairs. Returns written paths."""
        written: list[Path] = []
        for topic, content in items:
            written.append(self.ingest(topic, content))
        return written

    # ── Helpers ──

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ─── Snippet extraction ─────────────────────────────────────────────


def _snippet_around(text: str, needle: str, *, width: int = 80) -> str:
    """Return a short excerpt centered on the first ``needle`` match."""
    lower = text.lower()
    i = lower.find(needle.lower())
    if i == -1:
        return text[:width].strip()
    start = max(0, i - width // 2)
    end = min(len(text), i + len(needle) + width // 2)
    excerpt = text[start:end].replace("\n", " ").strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{excerpt}{suffix}"
