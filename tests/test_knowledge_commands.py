"""Tests for the /knowledge slash command handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.memory.commands import (
    handle_delete,
    handle_ingest,
    handle_list,
    handle_query,
    handle_show,
    handle_slash_knowledge,
)
from mouse.memory.semantic import SemanticMemory


@pytest.fixture
def mem(tmp_path: Path) -> SemanticMemory:
    return SemanticMemory(tmp_path / "semantic")


# ─── Individual handlers ────────────────────────────────────────────


def test_ingest_writes_topic(mem: SemanticMemory):
    out = handle_ingest(mem, "auth Session cookies are HTTP-only")
    assert "Ingested" in out
    assert mem.exists("auth")
    assert "Session cookies" in mem.get("auth")


def test_ingest_usage_when_missing_content(mem: SemanticMemory):
    assert "Usage" in handle_ingest(mem, "")
    assert "Usage" in handle_ingest(mem, "auth")


def test_query_shows_hits(mem: SemanticMemory):
    mem.ingest("auth", "uses OAuth2 and JWTs")
    out = handle_query(mem, "jwt")
    assert "auth" in out
    assert "Matches" in out


def test_query_no_hits(mem: SemanticMemory):
    mem.ingest("auth", "some content")
    out = handle_query(mem, "xyzzy")
    assert "No knowledge matches" in out


def test_query_usage_when_empty(mem: SemanticMemory):
    assert "Usage" in handle_query(mem, "")


def test_list_empty(mem: SemanticMemory):
    assert "No knowledge topics" in handle_list(mem)


def test_list_shows_topics(mem: SemanticMemory):
    mem.ingest("auth", "a")
    mem.ingest("deploy", "b")
    out = handle_list(mem)
    assert "auth" in out
    assert "deploy" in out
    assert "(2)" in out


def test_show_returns_raw_markdown(mem: SemanticMemory):
    mem.ingest("auth", "the auth content", title="Auth Flow")
    out = handle_show(mem, "auth")
    assert out.startswith("# Auth Flow")
    assert "the auth content" in out


def test_show_missing_topic(mem: SemanticMemory):
    assert "No topic" in handle_show(mem, "nope")


def test_show_usage_when_empty(mem: SemanticMemory):
    assert "Usage" in handle_show(mem, "")


def test_delete_removes_topic(mem: SemanticMemory):
    mem.ingest("auth", "stuff")
    out = handle_delete(mem, "auth")
    assert "Deleted" in out
    assert not mem.exists("auth")


def test_delete_missing_topic(mem: SemanticMemory):
    assert "No topic" in handle_delete(mem, "nope")


def test_delete_usage_when_empty(mem: SemanticMemory):
    assert "Usage" in handle_delete(mem, "")


def test_show_ignores_trailing_args(mem: SemanticMemory):
    """Regression: `show auth extra junk` should show topic `auth`,
    not slugify the whole tail to `auth-extra-junk` and miss."""
    mem.ingest("auth", "the auth content", title="Auth")
    out = handle_show(mem, "auth extra junk")
    assert "the auth content" in out


def test_delete_ignores_trailing_args(mem: SemanticMemory):
    mem.ingest("auth", "stuff")
    out = handle_delete(mem, "auth please")
    assert "Deleted" in out
    assert not mem.exists("auth")


# ─── Dispatcher ──────────────────────────────────────────────────────


def test_dispatcher_returns_none_for_other_commands(mem: SemanticMemory):
    assert handle_slash_knowledge("/sessions", mem) is None
    assert handle_slash_knowledge("something", mem) is None


def test_dispatcher_bare_knowledge_shows_usage(mem: SemanticMemory):
    out = handle_slash_knowledge("/knowledge", mem)
    assert out is not None
    assert "Usage" in out


def test_dispatcher_routes_ingest(mem: SemanticMemory):
    out = handle_slash_knowledge("/knowledge ingest auth some content here", mem)
    assert "Ingested" in out
    assert mem.exists("auth")


def test_dispatcher_routes_query(mem: SemanticMemory):
    mem.ingest("auth", "uses OAuth2")
    out = handle_slash_knowledge("/knowledge query oauth2", mem)
    assert "auth" in out


def test_dispatcher_routes_list(mem: SemanticMemory):
    mem.ingest("auth", "x")
    out = handle_slash_knowledge("/knowledge list", mem)
    assert "auth" in out


def test_dispatcher_routes_show(mem: SemanticMemory):
    mem.ingest("auth", "secret content")
    out = handle_slash_knowledge("/knowledge show auth", mem)
    assert "secret content" in out


def test_dispatcher_routes_delete(mem: SemanticMemory):
    mem.ingest("auth", "x")
    out = handle_slash_knowledge("/knowledge delete auth", mem)
    assert "Deleted" in out


def test_dispatcher_unknown_subcommand(mem: SemanticMemory):
    out = handle_slash_knowledge("/knowledge frobnicate", mem)
    assert out is not None
    assert "Unknown" in out
    assert "Usage" in out


def test_dispatcher_is_case_insensitive_on_head(mem: SemanticMemory):
    out = handle_slash_knowledge("/KNOWLEDGE list", mem)
    assert out is not None
    assert "No knowledge topics" in out
