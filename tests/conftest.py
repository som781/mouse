"""Shared fixtures and fakes for the mouse test suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


# ─── Fake litellm response shapes ────────────────────────────────────


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FakeFunction:
    name: str
    arguments: str  # JSON-encoded


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


class FakeMessage:
    """Mimics litellm's assistant message just enough for the agent loop."""

    def __init__(self, content: str | None = None, tool_calls: list[FakeToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (self.tool_calls or [])
            ],
        }


@dataclass
class FakeChoice:
    message: FakeMessage


class FakeResponse:
    def __init__(self, message: FakeMessage, usage: FakeUsage | None = None):
        self.choices = [FakeChoice(message=message)]
        self.usage = usage or FakeUsage(prompt_tokens=1, completion_tokens=1)


# ─── Scriptable fake LLM client ──────────────────────────────────────


class FakeLLMClient:
    """An LLMClient stand-in that returns scripted FakeResponses in order.

    Once the script is exhausted it returns the last response forever
    (a sentinel "done" message), so misbehaving loops can't hang.
    """

    def __init__(self, model: str, responses: list[FakeResponse]):
        self.model = model
        self.max_retries = 0
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def set_model(self, model: str) -> None:
        self.model = model

    def complete(self, *, messages, tools=None, tool_choice=None, max_tokens=None):
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
        })
        if not self.responses:
            return FakeResponse(FakeMessage(content="(done)"))
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


@pytest.fixture
def make_llm():
    """Factory to build a FakeLLMClient with a list of canned responses."""
    def _make(*responses: FakeResponse, model: str = "fake-model") -> FakeLLMClient:
        return FakeLLMClient(model, list(responses))
    return _make
