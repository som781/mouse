"""Tests for MCPConnection health tracking, transport-error detection,
and configurable per-call timeout.

These tests don't spin up a real MCP server — the session object is
stubbed so each test can drive call_tool() along a specific failure
path and assert how the harness responds.
"""

from __future__ import annotations

import asyncio

import pytest

from mouse.mcp.connection import (
    MCPConnection,
    _looks_like_transport_error,
)


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeSession:
    """Minimal stand-in for mcp.ClientSession. Each instance can be
    configured to succeed, raise a transport error, or raise a
    domain-level error on the next call_tool invocation."""

    def __init__(self) -> None:
        self.next_behavior: str = "ok"
        self.call_log: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name: str, arguments: dict):
        self.call_log.append((tool_name, arguments))
        if self.next_behavior == "ok":
            return _FakeResult(f"{tool_name}:{arguments}")
        if self.next_behavior == "transport":
            raise RuntimeError("session is closed")
        if self.next_behavior == "domain":
            raise ValueError("bad arguments — field `foo` required")
        raise AssertionError(f"unknown behavior {self.next_behavior}")


# ─── _looks_like_transport_error ─────────────────────────────────────


def test_transport_error_matches_known_class_name():
    class ClosedResourceError(Exception):
        pass

    assert _looks_like_transport_error(ClosedResourceError("anything")) is True


def test_transport_error_matches_substring_in_message():
    assert _looks_like_transport_error(RuntimeError("Connection closed")) is True
    assert _looks_like_transport_error(RuntimeError("Broken pipe")) is True


def test_domain_error_is_not_classified_as_transport():
    assert _looks_like_transport_error(ValueError("bad field")) is False
    assert _looks_like_transport_error(RuntimeError("")) is False


# ─── call_timeout property ───────────────────────────────────────────


def test_call_timeout_defaults_to_class_constant():
    conn = MCPConnection("s", {"command": "echo"})
    assert conn.call_timeout == MCPConnection.DEFAULT_CALL_TIMEOUT


def test_call_timeout_honors_server_config_override():
    conn = MCPConnection("s", {"command": "echo", "call_timeout": 5})
    assert conn.call_timeout == 5.0


def test_call_timeout_ignores_garbage_override():
    conn = MCPConnection("s", {"command": "echo", "call_timeout": "nope"})
    assert conn.call_timeout == MCPConnection.DEFAULT_CALL_TIMEOUT


# ─── call_tool + health tracking ─────────────────────────────────────


def test_call_tool_returns_not_connected_when_session_absent_and_reconnect_fails(monkeypatch):
    conn = MCPConnection("s", {"command": "echo"})
    # healthy defaults to False and reconnect will try self.connect()
    # which we stub to return False to simulate a still-dead server.
    async def fail_connect():
        return False
    monkeypatch.setattr(conn, "connect", fail_connect)

    out = asyncio.run(conn.call_tool("anything", {}))
    assert "not connected" in out


def test_call_tool_flags_unhealthy_on_transport_exception():
    conn = MCPConnection("s", {"command": "echo"})
    fake = _FakeSession()
    fake.next_behavior = "transport"
    conn.session = fake
    conn.healthy = True

    out = asyncio.run(conn.call_tool("list_things", {"x": 1}))
    assert out.startswith("ERROR:")
    assert conn.healthy is False  # next call will trigger reconnect


def test_call_tool_keeps_healthy_on_domain_exception():
    """A ValueError-shaped failure is the server rejecting the args,
    not a dead stream. The harness must not tear down a healthy
    session over a domain-level error."""
    conn = MCPConnection("s", {"command": "echo"})
    fake = _FakeSession()
    fake.next_behavior = "domain"
    conn.session = fake
    conn.healthy = True

    out = asyncio.run(conn.call_tool("list_things", {}))
    assert out.startswith("ERROR:")
    assert conn.healthy is True  # session still considered live


def test_call_tool_reconnects_lazily_when_previously_unhealthy(monkeypatch):
    """If the prior call flagged the session unhealthy, the next
    dispatch must call reconnect() before forwarding."""
    conn = MCPConnection("s", {"command": "echo"})
    reconnect_calls = {"n": 0}

    async def fake_reconnect():
        reconnect_calls["n"] += 1
        conn.session = _FakeSession()
        conn.healthy = True
        return True

    monkeypatch.setattr(conn, "reconnect", fake_reconnect)

    # Start unhealthy with no session — call_tool should reconnect.
    conn.healthy = False
    conn.session = None

    out = asyncio.run(conn.call_tool("ping", {"msg": "hi"}))
    assert reconnect_calls["n"] == 1
    assert "ping" in out


def test_call_tool_successful_dispatch_returns_text():
    conn = MCPConnection("s", {"command": "echo"})
    conn.session = _FakeSession()
    conn.healthy = True

    out = asyncio.run(conn.call_tool("ping", {"msg": "hi"}))
    assert "ping" in out
    assert conn.healthy is True
