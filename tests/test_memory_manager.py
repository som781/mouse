"""Tests for MemoryManager and the memory injector."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.memory.episodic import Observation
from mouse.memory.injector import (
    InjectionContext,
    _flatten,
    build_episodic_section,
    build_memory_context,
    build_preferences_section,
    build_semantic_section,
)
from mouse.memory.manager import MemoryManager


@pytest.fixture
def mem(tmp_path: Path) -> MemoryManager:
    return MemoryManager(tmp_path / "memory")


# ─── MemoryManager ──────────────────────────────────────────────────


def test_manager_creates_all_subdirs(tmp_path: Path):
    base = tmp_path / "nested" / "memory"
    m = MemoryManager(base)
    assert (base / "episodic").is_dir()
    assert (base / "semantic").is_dir()
    # preferences file is created lazily on first write, but its
    # parent is the base dir which must exist.
    assert base.is_dir()
    # The three stores point at the right places.
    assert m.episodic.base_dir == base / "episodic"
    assert m.semantic.base_dir == base / "semantic"
    assert m.preferences.path == base / "preferences.json"


def test_manager_paths_returns_mapping(mem: MemoryManager):
    paths = mem.paths()
    assert set(paths.keys()) == {"base", "episodic", "preferences", "semantic"}


def test_manager_layers_are_usable(mem: MemoryManager):
    """Sanity: each layer can be written and read through the manager."""
    mem.episodic.record(Observation("hello"), session_id="s1", session_title="t",
                        date="2026-04-07")
    mem.preferences.set("style.tone", "terse")
    mem.semantic.ingest("auth", "CORS first")

    assert "hello" in mem.episodic.read_day("2026-04-07")
    assert mem.preferences.get("style.tone") == "terse"
    assert "CORS first" in mem.semantic.get("auth")


# ─── Injector: section builders ─────────────────────────────────────


def test_episodic_section_empty_when_no_observations(mem: MemoryManager):
    assert build_episodic_section(mem) == ""


def test_episodic_section_includes_recent(mem: MemoryManager):
    mem.episodic.record(Observation("found cors issue"), session_id="s1",
                        session_title="a", date="2026-04-07")
    out = build_episodic_section(mem, days=3)
    assert "Recent observations" in out
    assert "cors issue" in out


def test_episodic_section_respects_char_budget(mem: MemoryManager):
    long = "x" * 5000
    mem.episodic.record(Observation(long), session_id="s1",
                        session_title="a", date="2026-04-07")
    out = build_episodic_section(mem, days=1, max_chars=200)
    # The header line + truncated content.
    assert "…" in out
    assert len(out) < 500


def test_preferences_section_empty_when_no_prefs(mem: MemoryManager):
    assert build_preferences_section(mem) == ""


def test_preferences_section_flattens_nested_keys(mem: MemoryManager):
    mem.preferences.update({
        "coding.formatter": "ruff",
        "coding.language": "python",
        "style.tone": "terse",
    })
    out = build_preferences_section(mem)
    assert "User preferences" in out
    assert "coding.formatter: ruff" in out
    assert "coding.language: python" in out
    assert "style.tone: terse" in out


def test_preferences_section_formats_lists(mem: MemoryManager):
    mem.preferences.set("tools.avoid", ["curl", "wget"])
    out = build_preferences_section(mem)
    assert "tools.avoid: curl, wget" in out


def test_preferences_section_truncates_long_output(mem: MemoryManager):
    for i in range(200):
        mem.preferences.set(f"k{i}", "v" * 20)
    out = build_preferences_section(mem, max_chars=300)
    assert "…" in out
    assert len(out) < 500


def test_semantic_section_empty_without_hint(mem: MemoryManager):
    mem.semantic.ingest("auth", "anything")
    assert build_semantic_section(mem, "") == ""
    assert build_semantic_section(mem, "   ") == ""


def test_semantic_section_empty_when_no_hits(mem: MemoryManager):
    mem.semantic.ingest("auth", "oauth stuff")
    assert build_semantic_section(mem, "unrelated-term") == ""


def test_semantic_section_ranked_and_labeled(mem: MemoryManager):
    mem.semantic.ingest("auth", "the auth doc: uses OAUTH2", title="Auth Flow")
    mem.semantic.ingest("deploy", "unrelated content")
    out = build_semantic_section(mem, "oauth2")
    assert "Relevant knowledge" in out
    assert "Auth Flow" in out
    assert "auth" in out  # slug shown


# ─── Injector: top-level builder ────────────────────────────────────


def test_build_memory_context_empty_everywhere(mem: MemoryManager):
    ctx = build_memory_context(mem, hint="anything")
    assert isinstance(ctx, InjectionContext)
    assert ctx.is_empty
    assert ctx.markdown == ""


def test_build_memory_context_assembles_all_sections(mem: MemoryManager):
    mem.episodic.record(Observation("did auth thing"), session_id="s1",
                        session_title="a", date="2026-04-07")
    mem.preferences.set("style.tone", "terse")
    mem.semantic.ingest("auth", "uses OAUTH2 cookies", title="Auth")

    ctx = build_memory_context(mem, hint="oauth2")
    assert not ctx.is_empty
    assert ctx.markdown.startswith("# Memory context")
    assert "Recent observations" in ctx.markdown
    assert "User preferences" in ctx.markdown
    assert "Relevant knowledge" in ctx.markdown
    assert "did auth thing" in ctx.episodic
    assert "terse" in ctx.preferences
    assert "Auth" in ctx.semantic


def test_build_memory_context_skips_semantic_without_hint(mem: MemoryManager):
    mem.semantic.ingest("auth", "oauth2 stuff")
    mem.preferences.set("k", "v")
    ctx = build_memory_context(mem, hint="")
    assert "Relevant knowledge" not in ctx.markdown
    assert "User preferences" in ctx.markdown


def test_episodic_section_excludes_active_session(mem: MemoryManager):
    """The active session's observations are already in working memory;
    the injector must not echo them back as 'recent observations'."""
    mem.episodic.record(Observation("earlier-session note"),
                        session_id="prev", session_title="old work",
                        date="2026-04-08")
    mem.episodic.record(Observation("active-session note"),
                        session_id="active", session_title="current",
                        date="2026-04-08")
    out = build_episodic_section(mem, exclude_session="active")
    assert "earlier-session note" in out
    assert "active-session note" not in out
    assert "## Session: current (active)" not in out


def test_episodic_section_without_exclude_includes_everything(mem: MemoryManager):
    mem.episodic.record(Observation("active-session note"),
                        session_id="active", session_title="current",
                        date="2026-04-08")
    out = build_episodic_section(mem)
    assert "active-session note" in out


def test_build_memory_context_passes_exclude_session_through(mem: MemoryManager):
    mem.episodic.record(Observation("old"), session_id="prev",
                        session_title="a", date="2026-04-08")
    mem.episodic.record(Observation("live"), session_id="active",
                        session_title="b", date="2026-04-08")
    ctx = build_memory_context(mem, exclude_session="active")
    assert "old" in ctx.episodic
    assert "live" not in ctx.episodic


def test_episodic_section_exclude_when_only_active_session_present(mem: MemoryManager):
    """If the day contains ONLY the active session, stripping it
    should leave the section empty, not emit a dangling header."""
    mem.episodic.record(Observation("solo"), session_id="active",
                        session_title="x", date="2026-04-08")
    out = build_episodic_section(mem, exclude_session="active")
    assert out == ""


def test_build_memory_context_only_preferences(mem: MemoryManager):
    mem.preferences.set("style.tone", "terse")
    ctx = build_memory_context(mem)
    assert ctx.episodic == ""
    assert ctx.semantic == ""
    assert "User preferences" in ctx.markdown


# ─── Helpers ────────────────────────────────────────────────────────


def test_flatten_nested():
    assert _flatten({"a": {"b": {"c": 1}, "d": 2}, "e": 3}) == [
        ("a.b.c", 1),
        ("a.d", 2),
        ("e", 3),
    ]


def test_flatten_empty():
    assert _flatten({}) == []
