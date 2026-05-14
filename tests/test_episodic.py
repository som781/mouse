"""Tests for the episodic memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.memory.episodic import (
    EpisodicMemory,
    EpisodicMemoryError,
    Observation,
    _parse_session_ids,
    _session_header_bounds,
)


@pytest.fixture
def mem(tmp_path: Path) -> EpisodicMemory:
    return EpisodicMemory(tmp_path / "episodic")


# ─── Errors & construction ──────────────────────────────────────────


def test_error_is_mouse_error():
    assert issubclass(EpisodicMemoryError, MouseError)


def test_memory_creates_base_dir(tmp_path: Path):
    base = tmp_path / "nested" / "episodic"
    assert not base.exists()
    EpisodicMemory(base)
    assert base.is_dir()


# ─── Observation ────────────────────────────────────────────────────


def test_observation_renders_with_kind():
    obs = Observation("CORS middleware must be added before handlers", kind="discovered")
    assert obs.render() == "- Discovered: CORS middleware must be added before handlers"


def test_observation_default_kind_is_note():
    obs = Observation("production only")
    assert obs.render().startswith("- Note:")


def test_observation_strips_whitespace():
    obs = Observation("  padded  ", kind="fixed")
    assert obs.render() == "- Fixed: padded"


# ─── Header parsing ─────────────────────────────────────────────────


def test_parse_session_ids_empty():
    assert _parse_session_ids("") == []


def test_parse_session_ids_finds_multiple():
    content = """# 2026-04-07

## Session: first one (abc123)
- stuff

## Session: second one (def456)
- more stuff
"""
    assert _parse_session_ids(content) == ["abc123", "def456"]


def test_session_header_bounds_finds_block():
    content = """# 2026-04-07

## Session: a (abc)
- one
- two

## Session: b (def)
- three
"""
    bounds = _session_header_bounds(content, "abc")
    assert bounds is not None
    start, end = bounds
    assert content[start:end].startswith("## Session: a (abc)")
    assert "two" in content[start:end]
    # The block ends before the next ## header.
    assert "def" not in content[start:end]


def test_session_header_bounds_last_block_goes_to_eof():
    content = "# x\n\n## Session: only (zzz)\n- alone\n"
    bounds = _session_header_bounds(content, "zzz")
    assert bounds is not None
    start, end = bounds
    assert end == len(content)


def test_session_header_bounds_missing_returns_none():
    content = "# x\n\n## Session: other (abc)\n- stuff\n"
    assert _session_header_bounds(content, "nope") is None


# ─── Recording ──────────────────────────────────────────────────────


def test_record_creates_day_file_with_header(mem: EpisodicMemory):
    path = mem.record(
        Observation("first thing", kind="discovered"),
        session_id="s1",
        session_title="Initial session",
        date="2026-04-07",
    )
    assert path.exists()
    text = path.read_text()
    assert text.startswith("# 2026-04-07")
    assert "## Session: Initial session (s1)" in text
    assert "- Discovered: first thing" in text


def test_record_requires_session_id(mem: EpisodicMemory):
    with pytest.raises(EpisodicMemoryError, match="session_id"):
        mem.record(Observation("x"), session_id="", session_title="t")


def test_record_accepts_list_of_observations(mem: EpisodicMemory):
    mem.record(
        [
            Observation("one", kind="discovered"),
            Observation("two", kind="fixed"),
        ],
        session_id="s1",
        session_title="batch",
        date="2026-04-07",
    )
    text = mem.read_day("2026-04-07")
    assert "- Discovered: one" in text
    assert "- Fixed: two" in text


def test_record_no_observations_is_noop(mem: EpisodicMemory):
    # Empty iterable shouldn't create the file.
    path = mem.record([], session_id="s1", date="2026-04-07")
    assert not path.exists()


def test_record_same_session_appends_to_existing_block(mem: EpisodicMemory):
    mem.record(
        Observation("first"),
        session_id="s1", session_title="work", date="2026-04-07",
    )
    mem.record(
        Observation("second", kind="fixed"),
        session_id="s1", session_title="work", date="2026-04-07",
    )
    text = mem.read_day("2026-04-07")
    # Both observations live under a single session header.
    assert text.count("## Session: work (s1)") == 1
    assert text.index("first") < text.index("second")


def test_record_different_sessions_create_separate_blocks(mem: EpisodicMemory):
    mem.record(
        Observation("one"),
        session_id="s1", session_title="first", date="2026-04-07",
    )
    mem.record(
        Observation("two"),
        session_id="s2", session_title="second", date="2026-04-07",
    )
    text = mem.read_day("2026-04-07")
    assert "## Session: first (s1)" in text
    assert "## Session: second (s2)" in text


def test_record_appending_to_earlier_session_preserves_later_one(
    mem: EpisodicMemory,
):
    """Appending to session s1 must not clobber s2's block that came after."""
    mem.record(Observation("s1-initial"), session_id="s1", session_title="a",
               date="2026-04-07")
    mem.record(Observation("s2-initial"), session_id="s2", session_title="b",
               date="2026-04-07")
    mem.record(Observation("s1-followup"), session_id="s1", session_title="a",
               date="2026-04-07")

    text = mem.read_day("2026-04-07")
    assert text.count("## Session: a (s1)") == 1
    assert text.count("## Session: b (s2)") == 1
    assert "s1-initial" in text
    assert "s1-followup" in text
    assert "s2-initial" in text
    # s1's follow-up lives inside the s1 block, before s2's header.
    assert text.index("s1-followup") < text.index("## Session: b (s2)")


def test_record_ignores_non_observation_items_in_list(mem: EpisodicMemory):
    mem.record(
        [Observation("real"), "not an observation", None],  # type: ignore[list-item]
        session_id="s1", session_title="x", date="2026-04-07",
    )
    text = mem.read_day("2026-04-07")
    assert "real" in text
    assert "not an observation" not in text


# ─── Reading ────────────────────────────────────────────────────────


def test_read_day_missing_returns_empty_string(mem: EpisodicMemory):
    assert mem.read_day("1999-01-01") == ""


def test_list_days_returns_sorted_iso_dates(mem: EpisodicMemory):
    for d in ("2026-04-07", "2026-04-05", "2026-04-06"):
        mem.record(Observation("x"), session_id="s", session_title="t", date=d)
    assert mem.list_days() == ["2026-04-05", "2026-04-06", "2026-04-07"]


def test_list_days_skips_malformed_filenames(mem: EpisodicMemory, tmp_path: Path):
    # Drop an unrelated file in the directory.
    (mem.base_dir / "notes.md").write_text("irrelevant")
    (mem.base_dir / "README.md").write_text("also irrelevant")
    mem.record(Observation("x"), session_id="s", session_title="t", date="2026-04-07")
    assert mem.list_days() == ["2026-04-07"]


def test_recent_returns_newest_first_and_caps(mem: EpisodicMemory):
    for d in ("2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04"):
        mem.record(Observation(d), session_id="s", session_title="t", date=d)
    recent = mem.recent(days=2)
    assert [date for date, _ in recent] == ["2026-04-04", "2026-04-03"]


def test_recent_zero_days_returns_empty(mem: EpisodicMemory):
    mem.record(Observation("x"), session_id="s", session_title="t", date="2026-04-07")
    assert mem.recent(days=0) == []


def test_recent_skips_missing_files(mem: EpisodicMemory):
    # Only one file actually exists.
    mem.record(Observation("x"), session_id="s", session_title="t", date="2026-04-07")
    recent = mem.recent(days=5)
    assert len(recent) == 1


def test_session_observations_returns_only_matching_block(mem: EpisodicMemory):
    mem.record(Observation("alpha"), session_id="s1", session_title="a",
               date="2026-04-07")
    mem.record(Observation("beta"), session_id="s2", session_title="b",
               date="2026-04-07")
    block = mem.session_observations("s1", date="2026-04-07")
    assert "alpha" in block
    assert "beta" not in block
    assert block.startswith("## Session: a (s1)")


def test_session_observations_missing_returns_empty(mem: EpisodicMemory):
    assert mem.session_observations("nope", date="2026-04-07") == ""


# ─── Today / utility ────────────────────────────────────────────────


def test_today_returns_iso_date(mem: EpisodicMemory):
    today = mem.today()
    assert len(today) == 10 and today[4] == "-" and today[7] == "-"
