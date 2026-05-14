"""Tests for the tool output store and its fetch/search built-ins."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.tools import tool_output_store
from mouse.tools.builtin import (
    handle_fetch_tool_output,
    handle_list_tool_outputs,
    handle_search_tool_output,
    build_default_tools,
)


@pytest.fixture(autouse=True)
def _clear_store():
    """Isolate each test."""
    tool_output_store.configure(None)  # reset to memory backend
    tool_output_store.clear()
    yield
    tool_output_store.configure(None)
    tool_output_store.clear()


# ─── Store ───────────────────────────────────────────────────────────


def test_save_and_fetch_full():
    tool_output_store.save("call_1", "hello world")
    out = tool_output_store.fetch("call_1")
    assert "hello world" in out
    assert "chars 0:11 of 11" in out


def test_fetch_paginates():
    content = "x" * 10_000
    tool_output_store.save("call_2", content)
    slice1 = tool_output_store.fetch("call_2", offset=0, limit=4000)
    assert "chars 0:4000" in slice1
    assert "more chars" in slice1
    slice2 = tool_output_store.fetch("call_2", offset=4000, limit=4000)
    assert "chars 4000:8000" in slice2


def test_fetch_offset_past_end():
    tool_output_store.save("call_3", "short")
    out = tool_output_store.fetch("call_3", offset=999)
    assert "past end" in out


def test_fetch_unknown_id_returns_error():
    out = tool_output_store.fetch("does_not_exist")
    assert out.startswith("ERROR:")


def test_save_empty_id_is_ignored():
    tool_output_store.save("", "content")
    assert tool_output_store.size() == 0


# ─── Search ──────────────────────────────────────────────────────────


def test_search_finds_matches_with_context():
    content = "\n".join([
        "line one",
        "line two",
        "hit here",
        "line four",
        "line five",
    ])
    tool_output_store.save("call_s", content)
    out = tool_output_store.search("call_s", r"hit")
    assert "hit here" in out
    # Default context=1 includes neighbors.
    assert "line two" in out
    assert "line four" in out
    assert "1 matches" in out


def test_search_no_matches():
    tool_output_store.save("call_s", "nothing\nto\nsee\nhere")
    out = tool_output_store.search("call_s", r"xyzzy")
    assert "No matches" in out


def test_search_invalid_regex():
    tool_output_store.save("call_s", "abc")
    out = tool_output_store.search("call_s", r"(unclosed")
    assert out.startswith("ERROR:")


def test_search_caps_at_max_matches():
    content = "\n".join(f"match {i}" for i in range(50))
    tool_output_store.save("call_many", content)
    out = tool_output_store.search("call_many", r"match", max_matches=5)
    assert "and 45 more matches" in out


def test_search_overlapping_matches_merge_without_duplicates():
    """Consecutive matches with context=1 should merge into one block,
    not print overlapping lines twice or lose match markers."""
    content = "\n".join(["a", "hit1", "hit2", "hit3", "b"])
    tool_output_store.save("call_overlap", content)
    out = tool_output_store.search("call_overlap", r"hit", context=1)
    # Every match line must carry the '>' marker.
    assert out.count("> 2: hit1") == 1
    assert out.count("> 3: hit2") == 1
    assert out.count("> 4: hit3") == 1
    # Context lines should appear exactly once even though they're
    # adjacent to multiple matches.
    assert out.count("1: a") == 1
    assert out.count("5: b") == 1


def test_search_cap_counts_absorbed_matches_correctly():
    """When ``max_matches`` is smaller than the total hits, the "more"
    count must reflect exactly the number of matches beyond the cap —
    not be confused by overlap absorption."""
    # 10 consecutive "hit" lines; with context=1 they'd all merge into
    # one block. max_matches=3 means we render the first 3 matches, and
    # the other 7 are omitted.
    content = "\n".join(["hit"] * 10)
    tool_output_store.save("call_cap", content)
    out = tool_output_store.search("call_cap", r"hit", max_matches=3, context=1)
    assert "10 matches" in out   # header counts all hits
    assert "and 7 more matches" in out


def test_search_unknown_id():
    out = tool_output_store.search("nope", "pattern")
    assert out.startswith("ERROR:")


# ─── Built-in tool handlers ──────────────────────────────────────────


def test_fetch_handler_uses_store():
    tool_output_store.save("call_h", "data " * 100)
    result = handle_fetch_tool_output({"tool_call_id": "call_h", "limit": 50})
    assert "chars 0:50" in result


def test_search_handler_uses_store():
    tool_output_store.save("call_h", "alpha\nbeta\ngamma")
    result = handle_search_tool_output({
        "tool_call_id": "call_h",
        "pattern": "beta",
        "context": 0,
    })
    assert "beta" in result
    assert "alpha" not in result  # context=0 drops neighbors


# ─── Disk backend ────────────────────────────────────────────────────


def test_configure_enables_disk_backend(tmp_path: Path):
    tool_output_store.configure(tmp_path / "sess1")
    tool_output_store.save("call_d", "disk content")
    # Written under <session_dir>/tool_outputs/<tc_id>.txt
    p = tmp_path / "sess1" / "tool_outputs" / "call_d.txt"
    assert p.exists()
    assert p.read_text() == "disk content"


def test_fetch_reads_from_disk(tmp_path: Path):
    tool_output_store.configure(tmp_path / "sess2")
    tool_output_store.save("call_d", "hello from disk")
    out = tool_output_store.fetch("call_d")
    assert "hello from disk" in out
    assert "chars 0:15 of 15" in out


def test_search_reads_from_disk(tmp_path: Path):
    tool_output_store.configure(tmp_path / "sess3")
    tool_output_store.save("call_d", "alpha\nbeta\ngamma")
    out = tool_output_store.search("call_d", "beta", context=0)
    assert "beta" in out
    assert "alpha" not in out


def test_size_counts_files_on_disk(tmp_path: Path):
    tool_output_store.configure(tmp_path / "sess4")
    assert tool_output_store.size() == 0
    tool_output_store.save("a", "1")
    tool_output_store.save("b", "2")
    tool_output_store.save("c", "3")
    assert tool_output_store.size() == 3


def test_clear_removes_disk_files(tmp_path: Path):
    tool_output_store.configure(tmp_path / "sess5")
    tool_output_store.save("a", "one")
    tool_output_store.save("b", "two")
    tool_output_store.clear()
    assert tool_output_store.size() == 0
    assert tool_output_store.fetch("a").startswith("ERROR:")


def test_disk_backend_sanitizes_exotic_ids(tmp_path: Path):
    """A tool_call_id from a weird MCP server must not break the
    filename — slashes and dots in unexpected places get replaced."""
    tool_output_store.configure(tmp_path / "sess6")
    tool_output_store.save("weird/id:with*chars", "payload")
    # Fetch still works via the same id.
    out = tool_output_store.fetch("weird/id:with*chars")
    assert "payload" in out


def test_disk_backend_survives_new_configure(tmp_path: Path):
    """Reconfiguring to the same dir must see previously saved files."""
    tool_output_store.configure(tmp_path / "sess7")
    tool_output_store.save("call_x", "persistent")
    tool_output_store.configure(None)
    assert tool_output_store.fetch("call_x").startswith("ERROR:")
    # Re-point at the same dir and the data is back.
    tool_output_store.configure(tmp_path / "sess7")
    assert "persistent" in tool_output_store.fetch("call_x")


def test_empty_tc_id_still_ignored_on_disk(tmp_path: Path):
    tool_output_store.configure(tmp_path / "sess8")
    tool_output_store.save("", "junk")
    assert tool_output_store.size() == 0


# ─── Registration ────────────────────────────────────────────────────


def test_tools_registered_in_default_registry():
    reg = build_default_tools()
    assert "fetch_tool_output" in reg.list_names()
    assert "search_tool_output" in reg.list_names()
    assert "list_tool_outputs" in reg.list_names()
    # All three are SAFE (read-only) permission.
    assert reg.get("fetch_tool_output").permission.value == "safe"
    assert reg.get("search_tool_output").permission.value == "safe"
    assert reg.get("list_tool_outputs").permission.value == "safe"


# ─── Format-aware preview ───────────────────────────────────────────


def test_preview_json_list_with_dict_items():
    payload = '[{"id": 1, "title": "a", "status": "open"}, {"id": 2, "title": "b", "status": "done"}]'
    tool_output_store.save("call_p1", payload, tool_name="list_tasks", args={"org": "x"})
    meta = tool_output_store.get_meta("call_p1")
    assert meta is not None
    assert meta["preview"]["format"] == "json_list"
    assert meta["preview"]["len"] == 2
    assert "id" in meta["preview"]["sample_keys"]
    assert "title" in meta["preview"]["sample_keys"]


def test_preview_json_object():
    tool_output_store.save("call_p2", '{"name": "acme", "id": 7, "active": true}')
    meta = tool_output_store.get_meta("call_p2")
    assert meta["preview"]["format"] == "json_object"
    assert "name" in meta["preview"]["keys"]


def test_preview_ndjson():
    payload = '{"a": 1}\n{"a": 2}\n{"a": 3}'
    tool_output_store.save("call_p3", payload)
    meta = tool_output_store.get_meta("call_p3")
    assert meta["preview"]["format"] == "ndjson"
    assert meta["preview"]["lines"] == 3


def test_preview_plain_text():
    tool_output_store.save("call_p4", "line one\nline two\nline three")
    meta = tool_output_store.get_meta("call_p4")
    assert meta["preview"]["format"] == "text"
    assert meta["preview"]["lines"] == 3
    assert meta["preview"]["first_line"] == "line one"


def test_preview_error_payload():
    tool_output_store.save("call_p5", "ERROR: thing exploded")
    meta = tool_output_store.get_meta("call_p5")
    assert meta["preview"]["format"] == "error"


def test_preview_empty_payload():
    tool_output_store.save("call_p6", "")
    # Empty content + no tc_id rule: empty content is OK, only empty
    # tc_id is rejected. So this should still produce a manifest.
    meta = tool_output_store.get_meta("call_p6")
    assert meta is not None
    assert meta["preview"]["format"] == "empty"


# ─── Manifest sidecar on disk ───────────────────────────────────────


def test_save_writes_manifest_sidecar(tmp_path: Path):
    tool_output_store.configure(tmp_path / "manifest_sess")
    tool_output_store.save(
        "call_m1", '[{"id": 1}]', tool_name="list_tasks", args={"org": "x"}
    )
    sidecar = tmp_path / "manifest_sess" / "tool_outputs" / "call_m1.json"
    assert sidecar.exists()
    import json as _json
    meta = _json.loads(sidecar.read_text())
    assert meta["tool_name"] == "list_tasks"
    assert meta["args"] == {"org": "x"}
    assert meta["total_chars"] == len('[{"id": 1}]')
    assert meta["preview"]["format"] == "json_list"
    assert meta["path"].endswith("call_m1.txt")


def test_path_for_returns_disk_path(tmp_path: Path):
    tool_output_store.configure(tmp_path / "pf_sess")
    tool_output_store.save("call_m2", "data")
    p = tool_output_store.path_for("call_m2")
    assert p is not None
    assert p.endswith("call_m2.txt")


def test_path_for_returns_none_in_memory_mode():
    tool_output_store.save("call_m3", "data")
    assert tool_output_store.path_for("call_m3") is None


def test_size_ignores_manifest_sidecars(tmp_path: Path):
    """Sidecars must not double the count."""
    tool_output_store.configure(tmp_path / "size_sess")
    tool_output_store.save("a", "1")
    tool_output_store.save("b", "2")
    assert tool_output_store.size() == 2  # not 4


def test_clear_removes_sidecars_too(tmp_path: Path):
    tool_output_store.configure(tmp_path / "clear_sess")
    tool_output_store.save("call_x", "data")
    assert (tmp_path / "clear_sess" / "tool_outputs" / "call_x.json").exists()
    tool_output_store.clear()
    assert not (tmp_path / "clear_sess" / "tool_outputs" / "call_x.json").exists()


# ─── list_outputs / get_meta ────────────────────────────────────────


def test_list_outputs_newest_first():
    tool_output_store.save("call_a", "1", tool_name="alpha")
    tool_output_store.save("call_b", "2", tool_name="beta")
    tool_output_store.save("call_c", "3", tool_name="alpha")
    rows = tool_output_store.list_outputs()
    assert len(rows) == 3
    # Newest first; call_c was saved last.
    assert rows[0]["tc_id"] == "call_c"


def test_list_outputs_filters_by_tool_name():
    tool_output_store.save("call_a", "1", tool_name="alpha")
    tool_output_store.save("call_b", "2", tool_name="beta")
    tool_output_store.save("call_c", "3", tool_name="alpha")
    rows = tool_output_store.list_outputs(tool_name="alpha")
    names = {r["tc_id"] for r in rows}
    assert names == {"call_a", "call_c"}


def test_list_outputs_respects_limit():
    for i in range(10):
        tool_output_store.save(f"call_{i}", str(i), tool_name="t")
    rows = tool_output_store.list_outputs(limit=3)
    assert len(rows) == 3


def test_list_outputs_disk_backend(tmp_path: Path):
    tool_output_store.configure(tmp_path / "list_sess")
    tool_output_store.save("call_a", '[{"x": 1}]', tool_name="alpha")
    tool_output_store.save("call_b", "plain text", tool_name="beta")
    rows = tool_output_store.list_outputs()
    assert len(rows) == 2
    formats = {r["tool_name"]: r["preview"]["format"] for r in rows}
    assert formats["alpha"] == "json_list"
    assert formats["beta"] == "text"


# ─── list_tool_outputs builtin handler ──────────────────────────────


def test_handle_list_tool_outputs_empty():
    out = handle_list_tool_outputs({})
    assert "No saved tool outputs" in out


def test_handle_list_tool_outputs_renders_rows():
    tool_output_store.save(
        "call_h1",
        '[{"id": 1, "title": "a"}, {"id": 2, "title": "b"}]',
        tool_name="list_tasks",
        args={"org": "acme"},
    )
    out = handle_list_tool_outputs({})
    assert "list_tasks" in out
    assert "json_list[2]" in out
    assert "tool_call_id=call_h1" in out
    assert "org='acme'" in out


def test_handle_list_tool_outputs_filter():
    tool_output_store.save("call_h1", "x", tool_name="alpha")
    tool_output_store.save("call_h2", "y", tool_name="beta")
    out = handle_list_tool_outputs({"tool_name": "beta"})
    assert "call_h2" in out
    assert "call_h1" not in out


# ─── Truncation marker carries the locator ─────────────────────────


def test_truncation_marker_includes_tc_id_and_path(tmp_path: Path):
    """The agent's _truncate_tool_result should embed both handles
    when the disk backend is active."""
    from mouse.engine.agent import _truncate_tool_result, MAX_TOOL_RESULT_CHARS
    big = "x" * (MAX_TOOL_RESULT_CHARS + 5_000)
    out = _truncate_tool_result(big, tc_id="call_xyz", path="/abs/foo.txt")
    assert "tool_call_id=call_xyz" in out
    assert "saved at /abs/foo.txt" in out
    assert "fetch_tool_output" in out
    assert "list_tool_outputs" in out


def test_truncation_marker_no_locator_when_none():
    from mouse.engine.agent import _truncate_tool_result, MAX_TOOL_RESULT_CHARS
    big = "x" * (MAX_TOOL_RESULT_CHARS + 100)
    out = _truncate_tool_result(big)
    # Marker still present, just without the locator parens.
    assert "truncated" in out
    assert "tool_call_id=" not in out
