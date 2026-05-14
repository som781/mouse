"""Tests for the append-only JSONL transcript writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.sessions.transcript import (
    TranscriptError,
    TranscriptWriter,
    iter_transcript,
    read_transcript,
    transcript_filename,
)


# ─── Filename ────────────────────────────────────────────────────────


def test_filename_format():
    name = transcript_filename("abc123", created_at=0)  # UTC epoch
    assert name == "1970-01-01_abc123.jsonl"


def test_filename_uses_current_time_by_default():
    name = transcript_filename("sess1")
    assert name.endswith("_sess1.jsonl")
    # YYYY-MM-DD is always 10 chars
    assert name[4] == "-" and name[7] == "-"


# ─── Errors ──────────────────────────────────────────────────────────


def test_transcript_error_is_mouse_error():
    assert issubclass(TranscriptError, MouseError)


# ─── Writer ──────────────────────────────────────────────────────────


def test_writer_creates_transcripts_dir_lazily(tmp_path: Path):
    w = TranscriptWriter("s1", tmp_path)
    try:
        assert (tmp_path / "transcripts").is_dir()
        # File isn't created until first append.
        assert not w.path.exists()
    finally:
        w.close()


def test_append_writes_one_line_per_call(tmp_path: Path):
    with TranscriptWriter("s1", tmp_path) as w:
        w.append({"role": "user", "content": "hi"})
        w.append({"role": "assistant", "content": "hello"})

    lines = [ln for ln in w.path.read_text().splitlines() if ln]
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["role"] == "user" and a["content"] == "hi"
    assert b["role"] == "assistant"
    # seq is monotonic and 1-based.
    assert a["seq"] == 1 and b["seq"] == 2
    # ts is attached.
    assert a["ts"] > 0 and b["ts"] >= a["ts"]


def test_append_rejects_non_dict(tmp_path: Path):
    with TranscriptWriter("s1", tmp_path) as w:
        with pytest.raises(TranscriptError, match="must be a dict"):
            w.append("not a dict")  # type: ignore[arg-type]


def test_append_rejects_unserializable_and_rolls_back_seq(tmp_path: Path):
    """A bad append must not bump seq — the next good append should
    still be 1, not 2."""
    with TranscriptWriter("s1", tmp_path) as w:
        class Blob:
            pass
        with pytest.raises(TranscriptError, match="not JSON-serializable"):
            w.append({"payload": Blob()})
        seq = w.append({"role": "user", "content": "ok"})
        assert seq == 1


def test_append_serializes_path_objects(tmp_path: Path):
    """The json default hook converts Path → str."""
    with TranscriptWriter("s1", tmp_path) as w:
        w.append({"file": Path("/tmp/x")})
    data = read_transcript(w.path)
    assert data[0]["file"] == "/tmp/x"


def test_reopen_continues_seq_numbering(tmp_path: Path):
    """Opening a writer on an existing transcript must pick up where
    the last seq left off."""
    with TranscriptWriter("s1", tmp_path) as w:
        assert w.append({"role": "user", "content": "one"}) == 1
        assert w.append({"role": "user", "content": "two"}) == 2

    with TranscriptWriter("s1", tmp_path) as w2:
        assert w2.append({"role": "user", "content": "three"}) == 3

    lines = [json.loads(l) for l in w2.path.read_text().splitlines() if l]
    assert [m["seq"] for m in lines] == [1, 2, 3]


def test_reopen_tolerates_corrupt_trailing_line(tmp_path: Path):
    """A partially-written (corrupt) line must not crash the reopen."""
    w = TranscriptWriter("s1", tmp_path)
    w.append({"role": "user", "content": "good"})
    w.close()
    # Append garbage as if a crash happened mid-write.
    with w.path.open("a") as f:
        f.write('{"seq": 99, bad json\n')

    w2 = TranscriptWriter("s1", tmp_path)
    try:
        # Last valid seq was 1, so the next append should be 2 — we
        # skipped the corrupt "seq: 99" line entirely.
        assert w2.append({"role": "user", "content": "more"}) == 2
    finally:
        w2.close()


def test_writer_context_manager_closes_file(tmp_path: Path):
    with TranscriptWriter("s1", tmp_path) as w:
        w.append({"role": "user", "content": "x"})
    # After __exit__, re-appending should still work (re-open lazily).
    w.append({"role": "user", "content": "y"})
    w.close()
    data = read_transcript(w.path)
    assert [m["content"] for m in data] == ["x", "y"]


# ─── Readers ─────────────────────────────────────────────────────────


def test_read_transcript_missing_file_returns_empty(tmp_path: Path):
    assert read_transcript(tmp_path / "nope.jsonl") == []


def test_read_transcript_skips_blank_and_malformed_lines(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text('\n{"seq":1,"role":"user"}\nnot json\n  \n{"seq":2,"role":"assistant"}\n')
    data = read_transcript(p)
    assert [m["seq"] for m in data] == [1, 2]


def test_iter_transcript_streams_records(tmp_path: Path):
    with TranscriptWriter("s1", tmp_path) as w:
        for i in range(5):
            w.append({"role": "user", "content": f"msg{i}"})
    seen = [m["content"] for m in iter_transcript(w.path)]
    assert seen == [f"msg{i}" for i in range(5)]


def test_iter_transcript_missing_file_yields_nothing(tmp_path: Path):
    assert list(iter_transcript(tmp_path / "missing.jsonl")) == []
