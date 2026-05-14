"""Tests for mouse.tools — registry, schema, and built-in handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse.tools.builtin import (
    build_default_tools,
    handle_read_file,
    handle_write_file,
    handle_list_directory,
)
from mouse.tools.registry import PermissionLevel, Tool, ToolRegistry


# ─── Registry ────────────────────────────────────────────────────────


def test_registry_register_and_get():
    r = ToolRegistry()
    t = Tool(name="x", description="d", parameters={}, handler=lambda a: "")
    r.register(t)
    assert r.get("x") is t
    assert r.get("missing") is None
    assert r.list_names() == ["x"]


def test_registry_overwrite_keeps_latest():
    r = ToolRegistry()
    r.register(Tool(name="x", description="a", parameters={}, handler=lambda a: ""))
    r.register(Tool(name="x", description="b", parameters={}, handler=lambda a: ""))
    assert r.get("x").description == "b"


def test_to_openai_schema_shape():
    t = Tool(
        name="echo",
        description="echo back",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=lambda a: a["x"],
    )
    s = t.to_openai_schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo"
    assert s["function"]["description"] == "echo back"
    assert s["function"]["parameters"]["properties"] == {"x": {"type": "string"}}


def test_default_tools_registered():
    r = build_default_tools()
    names = r.list_names()
    assert set(names) == {
        "bash", "read_file", "write_file",
        "search_files", "list_directory", "python",
        "fetch_tool_output", "search_tool_output", "list_tool_outputs",
    }
    # Permissions sanity check
    assert r.get("read_file").permission == PermissionLevel.SAFE
    assert r.get("bash").permission == PermissionLevel.ASK
    assert r.get("write_file").permission == PermissionLevel.ASK
    assert r.get("fetch_tool_output").permission == PermissionLevel.SAFE
    assert r.get("search_tool_output").permission == PermissionLevel.SAFE


# ─── Built-in handlers (filesystem ones — no subprocess needed) ──────


def test_read_file_round_trip(tmp_path: Path):
    p = tmp_path / "hello.txt"
    p.write_text("hi there")
    assert handle_read_file({"path": str(p)}) == "hi there"


def test_read_file_missing_returns_error_string():
    out = handle_read_file({"path": "/nope/nada/nothing.txt"})
    assert out.startswith("ERROR:")


def test_read_file_truncates_large_files(tmp_path: Path):
    p = tmp_path / "big.txt"
    p.write_text("\n".join(f"line {i}" for i in range(800)))
    out = handle_read_file({"path": str(p)})
    assert "lines omitted" in out
    assert "line 0" in out and "line 799" in out


def test_write_file_creates_parent_dirs(tmp_path: Path):
    target = tmp_path / "deeply" / "nested" / "out.txt"
    msg = handle_write_file({"path": str(target), "content": "data"})
    assert target.read_text() == "data"
    assert "Wrote" in msg


def test_list_directory_returns_some_paths(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    out = handle_list_directory({"path": str(tmp_path)})
    assert "a.txt" in out and "b.txt" in out
