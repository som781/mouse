"""Tool, PermissionLevel, ToolRegistry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class PermissionLevel(Enum):
    """How much trust a tool requires."""
    SAFE = "safe"        # Auto-approve (read-only ops)
    ASK  = "ask"         # Ask user before executing
    DENY = "deny"        # Block entirely


@dataclass
class Tool:
    """A single tool the agent can use."""
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    permission: PermissionLevel = PermissionLevel.ASK

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry of all tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        """All registered tools in insertion order."""
        return list(self._tools.values())

    def all_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())
