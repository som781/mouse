
"""MCPManager — multi-server lifecycle on a background event loop."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

from mouse.term import C
from mouse.logging import get_logger
from mouse.project import PROJECT_ROOT
from mouse.mcp.connection import MCPConnection
from mouse.tools.registry import PermissionLevel, Tool, ToolRegistry

log = get_logger("mouse.mcp")


class MCPManager:
    """
    Manages multiple MCP server connections.
    Runs a background event loop to keep MCP sessions alive
    so tool calls can be made from synchronous code.
    """

    def __init__(self):
        self.connections: dict[str, MCPConnection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _start_background_loop(self):
        """Start a background event loop for MCP sessions."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run_async(self, coro):
        """Run an async coroutine on the background loop and wait for result."""
        if not self._loop or not self._loop.is_running():
            self._start_background_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    async def load_from_config(self, config_path: str) -> int:
        """Load MCP servers from a JSON config file. Returns count of tools loaded.

        Supports two formats:

        Format 1 (array style):
            {"servers": [{"name": "x", "url": "..."}, ...]}

        Format 2 (Claude Desktop / Claude Code style):
            {"mcpServers": {"server-name": {"url": "..."}, ...}}
        """
        if not os.path.exists(config_path):
            print(C.styled(f"  ⚠  MCP config not found: {config_path}", C.YELLOW))
            return 0

        with open(config_path) as f:
            config = json.load(f)

        # Normalize both formats into a list of {name, ...config}
        servers: list[dict] = []

        if "servers" in config:
            # Format 1: array style
            servers = config["servers"]
        elif "mcpServers" in config:
            # Format 2: Claude Desktop style {"mcpServers": {"name": {config}}}
            for name, server_conf in config["mcpServers"].items():
                entry = dict(server_conf)  # copy
                entry["name"] = name
                # Normalize: "transport": "http" + "url" → just "url"
                if entry.get("transport") == "http" and "url" in entry:
                    pass  # url is already set, connect() will handle it
                servers.append(entry)
        else:
            print(C.styled("  ⚠  MCP config has no 'servers' or 'mcpServers' key", C.YELLOW))
            return 0

        if not servers:
            return 0

        total_tools = 0
        for server_config in servers:
            name = server_config.get("name", "unnamed")
            print(C.styled(f"  🔌 Connecting to MCP server: {name}...", C.CYAN))

            try:
                conn = MCPConnection(name, server_config)
                if await conn.connect():
                    self.connections[name] = conn
                    tool_count = len(conn.tools)
                    total_tools += tool_count
                    tool_names = [t.name for t in conn.tools]
                    print(C.styled(f"  ✓ {name}: {tool_count} tools → {', '.join(tool_names)}", C.GREEN))
                else:
                    print(C.styled(f"  ✗ {name}: failed to connect", C.RED))
            except Exception as e:
                err_msg = str(e) or type(e).__name__
                print(C.styled(f"  ✗ {name}: {err_msg}", C.RED))
                print(C.styled("    Skipping this server, continuing with others...", C.DIM))

        return total_tools

    def register_tools(self, registry: ToolRegistry):
        """Register all MCP tools into the agent's ToolRegistry."""

        for server_name, conn in self.connections.items():
            for mcp_tool in conn.tools:
                # Clean up tool name: replace hyphens with underscores for valid function names
                safe_server = server_name.replace("-", "_").replace(" ", "_")
                tool_name = f"mcp_{safe_server}_{mcp_tool.name}"
                description = mcp_tool.description or f"MCP tool from {server_name}"

                # Build parameters schema from MCP tool input schema
                if mcp_tool.inputSchema:
                    parameters = mcp_tool.inputSchema
                else:
                    parameters = {"type": "object", "properties": {}}

                # Create a handler that calls the MCP server. The
                # per-call timeout is read from the connection each
                # invocation so a config reload (future work) can
                # retune timeouts without rebuilding the registry.
                def make_handler(connection: MCPConnection, tool_nm: str):
                    def handler(args: dict) -> str:
                        if self._loop and self._loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                connection.call_tool(tool_nm, args), self._loop
                            )
                            try:
                                return future.result(timeout=connection.call_timeout)
                            except TimeoutError:
                                # Cancel the in-flight coroutine so
                                # the background loop isn't still
                                # waiting on a hung server; mark the
                                # session unhealthy so the next call
                                # rebuilds it lazily.
                                future.cancel()
                                connection.healthy = False
                                return (
                                    f"ERROR: MCP tool call timed out after "
                                    f"{connection.call_timeout:.1f}s — session will "
                                    f"reconnect on next dispatch."
                                )
                        # Fallback: create a fresh loop (may not work
                        # with all transports, but better than crashing
                        # a call when the background loop has stopped).
                        return asyncio.run(connection.call_tool(tool_nm, args))
                    return handler

                registry.register(Tool(
                    name=tool_name,
                    description=f"[MCP:{server_name}] {description}",
                    parameters=parameters,
                    handler=make_handler(conn, mcp_tool.name),
                    permission=PermissionLevel.ASK,
                ))

    async def disconnect_all(self):
        """Disconnect all MCP servers."""
        for name, conn in self.connections.items():
            await conn.disconnect()
            print(C.styled(f"  🔌 Disconnected: {name}", C.DIM))

    def shutdown(self):
        """Synchronous shutdown — disconnects all servers and stops background loop."""
        if self._loop and self._loop.is_running():
            try:
                self._run_async(self.disconnect_all())
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=3)


def load_mcp_servers(registry: ToolRegistry) -> MCPManager | None:
    """
    Load MCP servers from config and register their tools.
    Config file path is read from MCP_CONFIG env var, or auto-detected in project root.
    """

    # Check if mcp package is installed
    try:
        import mcp  # noqa: F401
    except ImportError:
        print(C.styled("  ⚠  MCP SDK not installed. To enable MCP: pip install mcp", C.YELLOW))
        return None

    config_path = os.environ.get("MCP_CONFIG", "")

    # Auto-detect config file in project root and common subdirectories
    if not config_path:
        candidates = [
            "mcp.json",
            "mcp_servers.json",
            ".mcp.json",
            "mcp_config.json",
            os.path.join(".mcp", "config.json"),
            os.path.join(".cursor", "mcp.json"),       # Cursor format
            os.path.join(".claude", "mcp_servers.json"),  # Claude Code format
            os.path.join("scripts", "mcp.json"),        # scripts/ subfolder
            os.path.join("scripts", "mcp_servers.json"),
            os.path.join("bin", "mcp.json"),
            os.path.join("config", "mcp.json"),
        ]

        # Also check the directory where THIS script lives
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else ""
        if script_dir and script_dir != PROJECT_ROOT:
            for name in ["mcp.json", "mcp_servers.json", ".mcp.json"]:
                candidates.append(os.path.join(script_dir, name))

        for candidate in candidates:
            # Handle both absolute and relative paths
            if os.path.isabs(candidate):
                full_path = candidate
            else:
                full_path = os.path.join(PROJECT_ROOT, candidate)
            if os.path.exists(full_path):
                config_path = full_path
                break

    if not config_path:
        print(C.styled("  ℹ  No MCP config found. Looked for: mcp.json, mcp_servers.json, .mcp.json in project root", C.DIM))
        print(C.styled("     Set MCP_CONFIG=/path/to/config.json or create one in your project root.", C.DIM))
        return None

    print(C.styled(f"  📄 MCP config: {config_path}", C.CYAN))

    # Show the parsed config for debugging
    try:
        with open(config_path) as f:
            raw_config = json.load(f)

        if "mcpServers" in raw_config:
            server_names = list(raw_config["mcpServers"].keys())
            print(C.styled("     Format: Claude Desktop (mcpServers)", C.DIM))
        elif "servers" in raw_config:
            server_names = [s.get("name", "unnamed") for s in raw_config["servers"]]
            print(C.styled("     Format: Array (servers)", C.DIM))
        else:
            print(C.styled("  ✗ Config has no 'mcpServers' or 'servers' key", C.RED))
            return None

        print(C.styled(f"     Servers found: {', '.join(server_names)}", C.DIM))
    except json.JSONDecodeError as e:
        print(C.styled(f"  ✗ Invalid JSON in {config_path}: {e}", C.RED))
        return None

    manager = MCPManager()
    manager._start_background_loop()

    try:
        tool_count = manager._run_async(manager.load_from_config(config_path))
        if tool_count > 0:
            manager.register_tools(registry)
            print(C.styled(f"  ✓ Loaded {tool_count} MCP tools total", C.GREEN, C.BOLD))
        else:
            print(C.styled("  ⚠  Connected but no tools discovered", C.YELLOW))
        return manager
    except Exception as e:
        print(C.styled(f"  ✗ MCP loading failed: {e}", C.RED))
        import traceback
        traceback.print_exc()
        return None
