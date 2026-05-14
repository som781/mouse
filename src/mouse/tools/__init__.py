"""Tool registry, built-in tools, and permission checks."""

from mouse.tools.registry import PermissionLevel, Tool, ToolRegistry
from mouse.tools.permissions import PermissionManager, DANGEROUS_PATTERNS
from mouse.tools.builtin import build_default_tools

__all__ = [
    "PermissionLevel",
    "Tool",
    "ToolRegistry",
    "PermissionManager",
    "DANGEROUS_PATTERNS",
    "build_default_tools",
]
