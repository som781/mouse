"""Tests for the semantic-wiki memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.memory.semantic import (
    Hit,
    SemanticMemory,
    SemanticMemoryError,
    _snippet_around,
    slugify,
)


@pytest.fixture
def mem(tmp_path: Path) -> SemanticMemory:
    return SemanticMemory(tmp_path / "semantic")


# ─── Errors & construction ──────────────────────────────────────────


def test_error_is_mouse_error():
    assert issubclass(SemanticMemoryError, MouseError)


def test_creates_base_dir(tmp_path: Path):
    base = tmp_path / "nested" / "semantic"
    assert not base.exists()
    SemanticMemory(base)
    assert base.is_dir()


# ─── Slugify ────────────────────────────────────────────────────────


def test_slugify_basic():
    assert slugify("Auth Flow") == "auth-flow"


def test_slugify_strips_punctuation():
    assert slugify("Data Pipeline!!!") == "data-pipeline"


def test_slugify_collapses_separators():
    assert slugify("deploy__pipeline--staging") == "deploy-pipeline-staging"


def test_slugify_empty_raises():
    with pytest.raises(SemanticMemoryError, match="empty slug"):
        slugify("!!!")


def test_slugify_non_string_raises():
    with pytest.raises(SemanticMemoryError, match="must be a string"):
        slugify(123)  # type: ignore[arg-type]


# ─── Ingest ─────────────────────────────────────────────────────────


def test_ingest_creates_new_topic_with_header(mem: SemanticMemory):
    path = mem.ingest("Auth", "Session cookies are HTTP-only.", title="Auth Flow")
    assert path.exists()
    text = path.read_text()
    assert text.startswith("# Auth Flow")
    assert "Session cookies are HTTP-only." in text
    assert "_ingested " in text


def test_ingest_appends_to_existing_topic(mem: SemanticMemory):
    mem.ingest("auth", "first note", timestamp="2026-04-01 00:00 UTC")
    mem.ingest("auth", "second note", timestamp="2026-04-02 00:00 UTC")
    text = mem.get("auth")
    assert text.count("---") == 1  # separator only on the second write
    assert text.index("first note") < text.index("second note")
    assert "2026-04-01" in text
    assert "2026-04-02" in text


def test_ingest_empty_content_raises(mem: SemanticMemory):
    with pytest.raises(SemanticMemoryError, match="non-empty"):
        mem.ingest("auth", "   ")


def test_ingest_uses_topic_as_default_title(mem: SemanticMemory):
    mem.ingest("Deployment", "stuff")
    assert mem.title_of("deployment") == "Deployment"


def test_ingest_many(mem: SemanticMemory):
    paths = mem.ingest_many([
        ("auth", "a"),
        ("deploy", "b"),
    ])
    assert len(paths) == 2
    assert set(mem.list_topics()) == {"auth", "deploy"}


# ─── Read ───────────────────────────────────────────────────────────


def test_get_missing_returns_empty_string(mem: SemanticMemory):
    assert mem.get("nope") == ""


def test_exists_reflects_state(mem: SemanticMemory):
    assert mem.exists("auth") is False
    mem.ingest("auth", "stuff")
    assert mem.exists("auth") is True


def test_title_of_falls_back_to_slug(mem: SemanticMemory, tmp_path: Path):
    # Write a file without an H1 header directly.
    (mem.base_dir / "weird.md").write_text("no header here", encoding="utf-8")
    assert mem.title_of("weird") == "weird"


def test_list_topics_is_sorted(mem: SemanticMemory):
    mem.ingest("zebra", "z")
    mem.ingest("apple", "a")
    mem.ingest("mango", "m")
    assert mem.list_topics() == ["apple", "mango", "zebra"]


def test_list_topics_ignores_non_md_files(mem: SemanticMemory):
    (mem.base_dir / "ignored.txt").write_text("x")
    mem.ingest("auth", "stuff")
    assert mem.list_topics() == ["auth"]


# ─── Delete ─────────────────────────────────────────────────────────


def test_delete_removes_topic(mem: SemanticMemory):
    mem.ingest("auth", "stuff")
    assert mem.delete("auth") is True
    assert mem.exists("auth") is False


def test_delete_missing_returns_false(mem: SemanticMemory):
    assert mem.delete("nope") is False


# ─── Query ──────────────────────────────────────────────────────────


def test_query_finds_matches(mem: SemanticMemory):
    mem.ingest("auth", "CORS middleware must be added before handlers.")
    mem.ingest("deploy", "We use Kubernetes on GKE.")
    hits = mem.query("CORS")
    assert len(hits) == 1
    assert isinstance(hits[0], Hit)
    assert hits[0].topic == "auth"
    assert "CORS" in hits[0].snippet


def test_query_ranks_by_match_count(mem: SemanticMemory):
    mem.ingest("heavy", "auth auth auth auth")
    mem.ingest("light", "auth once")
    hits = mem.query("auth")
    assert [h.topic for h in hits] == ["heavy", "light"]
    assert hits[0].score > hits[1].score


def test_query_is_case_insensitive(mem: SemanticMemory):
    mem.ingest("auth", "Uses OAUTH2")
    hits = mem.query("oauth2")
    assert len(hits) == 1


def test_query_empty_returns_empty(mem: SemanticMemory):
    mem.ingest("auth", "stuff")
    assert mem.query("") == []
    assert mem.query("   ") == []


def test_query_no_matches_returns_empty(mem: SemanticMemory):
    mem.ingest("auth", "stuff")
    assert mem.query("xyzzy") == []


def test_query_respects_limit(mem: SemanticMemory):
    for i in range(5):
        mem.ingest(f"topic{i}", "match match match")
    hits = mem.query("match", limit=2)
    assert len(hits) == 2


def test_query_tie_broken_by_slug_alphabetically(mem: SemanticMemory):
    mem.ingest("zeta", "hit")
    mem.ingest("alpha", "hit")
    hits = mem.query("hit")
    assert [h.topic for h in hits] == ["alpha", "zeta"]


# ─── Snippet helper ─────────────────────────────────────────────────


def test_snippet_around_centers_on_match():
    text = "a" * 100 + "NEEDLE" + "b" * 100
    snippet = _snippet_around(text, "needle", width=20)
    assert "NEEDLE" in snippet
    assert snippet.startswith("…")
    assert snippet.endswith("…")


def test_snippet_around_missing_returns_prefix():
    text = "some short note"
    snippet = _snippet_around(text, "absent", width=80)
    assert snippet == "some short note"
