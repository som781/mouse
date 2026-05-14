"""MCP (Model Context Protocol) integration."""

from mouse.mcp.connection import MCPConnection
from mouse.mcp.manager import MCPManager, load_mcp_servers

__all__ = ["MCPConnection", "MCPManager", "load_mcp_servers"]
