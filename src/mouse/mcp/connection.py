"""MCPConnection — single MCP server session (stdio/http/sse)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mouse.term import C


class MCPConnection:
    """
    Manages a connection to a single MCP server.
    Supports both stdio (local process) and HTTP (remote) transports.

    Health tracking: once a call raises a transport-level exception the
    session is presumed dead. The next :meth:`call_tool` invocation
    will tear down and rebuild the session before forwarding the call,
    so the caller sees a single late-stage retry instead of a cascade
    of ``MCP server not connected`` errors.
    """

    # Per-server timeout for a single tool dispatch. Overridable via
    # ``call_timeout`` in the server config. Connection / shutdown
    # timeouts are handled separately at the manager layer.
    DEFAULT_CALL_TIMEOUT = 60.0

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.session: Any = None
        self._stdio_context: Any = None
        self._session_context: Any = None
        self.tools: list[dict] = []
        self.healthy: bool = False

    @property
    def call_timeout(self) -> float:
        raw = self.config.get("call_timeout", self.DEFAULT_CALL_TIMEOUT)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return self.DEFAULT_CALL_TIMEOUT

    async def connect(self):
        """Connect to the MCP server and discover its tools."""
        try:
            from mcp import ClientSession, StdioServerParameters  # noqa: F401
            from mcp.client.stdio import stdio_client  # noqa: F401
        except ImportError:
            print(C.styled("  ✗ MCP SDK not installed. Run: pip install mcp", C.RED))
            return False

        try:
            if "url" in self.config:
                # HTTP/SSE transport — wrap with timeout
                ok = await asyncio.wait_for(self._connect_http(), timeout=15)
            else:
                # Stdio transport (local process) — wrap with timeout
                ok = await asyncio.wait_for(self._connect_stdio(), timeout=15)
            self.healthy = bool(ok)
            return ok
        except asyncio.TimeoutError:
            print(C.styled(f"  ✗ MCP '{self.name}' connection timed out (15s)", C.RED))
            self.healthy = False
            return False
        except ConnectionRefusedError:
            print(C.styled(f"  ✗ MCP '{self.name}' connection refused — is the server running?", C.RED))
            self.healthy = False
            return False
        except Exception as e:
            err_msg = str(e) or type(e).__name__
            print(C.styled(f"  ✗ MCP '{self.name}' connection failed: {err_msg}", C.RED))
            self.healthy = False
            return False

    async def _connect_stdio(self):
        """Connect to a local MCP server via stdio."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.config["command"],
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})},
        )

        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()

        # Discover tools
        result = await self.session.list_tools()
        self.tools = result.tools
        return True

    async def _connect_http(self):
        """Connect to a remote MCP server via HTTP (tries Streamable HTTP, then SSE)."""
        url = self.config["url"]
        headers = self.config.get("headers", {})

        # Try Streamable HTTP first (newer protocol)
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            self._stdio_context = streamablehttp_client(url, headers=headers)
            read_stream, write_stream, _ = await self._stdio_context.__aenter__()

            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()
            await self.session.initialize()

            result = await self.session.list_tools()
            self.tools = result.tools
            return True

        except Exception as e1:
            # Clean up failed attempt
            try:
                if self._session_context:
                    await self._session_context.__aexit__(None, None, None)
                if self._stdio_context:
                    await self._stdio_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_context = None
            self._stdio_context = None

            # Fall back to SSE transport
            try:
                from mcp import ClientSession
                from mcp.client.sse import sse_client

                print(C.styled("    ↳ Streamable HTTP failed, trying SSE...", C.DIM))

                self._stdio_context = sse_client(url, headers=headers)
                read_stream, write_stream = await self._stdio_context.__aenter__()

                self._session_context = ClientSession(read_stream, write_stream)
                self.session = await self._session_context.__aenter__()
                await self.session.initialize()

                result = await self.session.list_tools()
                self.tools = result.tools
                return True

            except Exception as e2:
                print(C.styled(f"    ✗ Streamable HTTP error: {e1}", C.RED))
                print(C.styled(f"    ✗ SSE error: {e2}", C.RED))
                return False

    async def reconnect(self) -> bool:
        """Tear down and rebuild the session. Called lazily before the
        next tool dispatch when the connection has been flagged
        unhealthy so a dead stdio subprocess or dropped HTTP session
        recovers without user intervention."""
        await self.disconnect()
        self.session = None
        self._session_context = None
        self._stdio_context = None
        return await self.connect()

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on this MCP server.

        If the session was previously marked unhealthy (a prior call
        raised a transport error), attempt one lazy reconnect before
        the dispatch. If the dispatch itself raises, mark the
        connection unhealthy so the next call reconnects — but do not
        retry inside this call, since the failure may be a legitimate
        domain-level error the caller needs to see verbatim.
        """
        if not self.healthy or not self.session:
            ok = await self.reconnect()
            if not ok or not self.session:
                return "ERROR: MCP server not connected"

        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            # Extract text from result content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "data"):
                    parts.append(f"[binary data: {len(block.data)} bytes]")
                else:
                    parts.append(str(block))
            return "\n".join(parts) or "(no output)"
        except Exception as e:
            err_str = str(e)

            # Handle "structured_content must be a dict or None. Got list: [...]"
            # The server returned valid data but in a format the SDK doesn't accept.
            # Extract the data from the error message itself.
            if "Got list:" in err_str:
                try:
                    import ast
                    import json
                    # Extract the list portion from the error string
                    list_start = err_str.index("Got list:") + len("Got list:")
                    raw_data = err_str[list_start:].strip()
                    parsed = ast.literal_eval(raw_data)
                    return json.dumps(parsed, indent=2, default=str)
                except Exception:
                    return raw_data if raw_data else f"ERROR: {err_str}"

            # Transport-level errors break the stream; flag the session
            # so the *next* dispatch triggers a reconnect. We don't
            # retry in-place because the exception may be domain-level
            # (bad args, server-side error) rather than transport-level.
            if _looks_like_transport_error(e):
                self.healthy = False

            return f"ERROR: MCP tool call failed: {err_str}"

    async def disconnect(self):
        """Clean up the connection."""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except Exception:
            pass
        self.healthy = False


# Transport error heuristics. The mcp SDK does not surface a single
# "connection is dead" exception type — dead stdio pipes, closed
# anyio streams, and broken SSE connections all surface as different
# classes with overlapping string signatures. We match on both class
# name and message substring so adding a new transport later doesn't
# require editing an allowlist.
_TRANSPORT_ERROR_CLASSES = {
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "ConnectionResetError",
    "BrokenPipeError",
    "IncompleteReadError",
}
_TRANSPORT_ERROR_SUBSTRINGS = (
    "closed resource",
    "broken pipe",
    "connection reset",
    "connection closed",
    "session is closed",
    "stream is closed",
    "eof",
)


def _looks_like_transport_error(exc: BaseException) -> bool:
    """Best-effort detection of a dead-stream failure vs a legitimate
    tool-level error. Harness-generic: no knowledge of any specific
    server or tool — just patterns that indicate the MCP channel
    itself has failed and the session should be rebuilt."""
    cls = type(exc).__name__
    if cls in _TRANSPORT_ERROR_CLASSES:
        return True
    msg = str(exc).lower()
    if not msg:
        return False
    return any(s in msg for s in _TRANSPORT_ERROR_SUBSTRINGS)
