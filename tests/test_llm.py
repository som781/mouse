"""Tests for mouse.llm — LLMClient retry behavior and Usage extraction."""

from __future__ import annotations

import pytest

from mouse import llm as llm_module
from mouse.errors import MouseError
from mouse.llm import LLMClient, LLMError, Usage


def test_llm_error_is_mouse_error():
    assert issubclass(LLMError, MouseError)


def test_set_model():
    c = LLMClient("a", max_retries=0)
    c.set_model("b")
    assert c.model == "b"


def test_usage_from_missing_attribute():
    class Empty:
        pass
    assert Usage.from_response(Empty()).total == 0


def test_usage_from_full_response():
    class U:
        prompt_tokens = 10
        completion_tokens = 5
    class R:
        usage = U()
    u = Usage.from_response(R())
    assert u.input_tokens == 10
    assert u.output_tokens == 5
    assert u.total == 15


def test_complete_succeeds_first_try(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()
    monkeypatch.setattr(llm_module.litellm, "completion", lambda **kw: sentinel)
    c = LLMClient("x", max_retries=2, retry_backoff_seconds=0)
    assert c.complete(messages=[]) is sentinel


def test_complete_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}
    sentinel = object()
    def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return sentinel
    monkeypatch.setattr(llm_module.litellm, "completion", flaky)
    c = LLMClient("x", max_retries=2, retry_backoff_seconds=0)
    assert c.complete(messages=[]) is sentinel
    assert calls["n"] == 3


def test_complete_raises_after_exhaustion(monkeypatch: pytest.MonkeyPatch):
    def always_fail(**_kw):
        raise RuntimeError("nope")
    monkeypatch.setattr(llm_module.litellm, "completion", always_fail)
    c = LLMClient("x", max_retries=2, retry_backoff_seconds=0)
    with pytest.raises(LLMError) as exc_info:
        c.complete(messages=[])
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, RuntimeError)


def test_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    """litellm.BadRequestError (incl. ContextWindowExceededError) must surface immediately."""
    BadRequest = getattr(llm_module.litellm, "BadRequestError", None)
    if BadRequest is None:
        pytest.skip("litellm.BadRequestError not available")
    calls = {"n": 0}
    def fail(**_kw):
        calls["n"] += 1
        raise BadRequest("context window exceeded", model="x", llm_provider="openai")
    monkeypatch.setattr(llm_module.litellm, "completion", fail)
    c = LLMClient("x", max_retries=5, retry_backoff_seconds=0)
    with pytest.raises(LLMError) as exc_info:
        c.complete(messages=[])
    assert calls["n"] == 1, "BadRequestError must not be retried"
    assert exc_info.value.attempts == 1


def test_context_window_name_match_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    """Defensive: catch ContextWindowExceededError even if class hierarchy changes."""
    class ContextWindowExceededError(Exception):
        pass
    calls = {"n": 0}
    def fail(**_kw):
        calls["n"] += 1
        raise ContextWindowExceededError("too big")
    monkeypatch.setattr(llm_module.litellm, "completion", fail)
    c = LLMClient("x", max_retries=5, retry_backoff_seconds=0)
    with pytest.raises(LLMError):
        c.complete(messages=[])
    assert calls["n"] == 1


def test_complete_omits_tools_when_none(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}
    def spy(**kw):
        captured.update(kw)
        class Resp: pass
        return Resp()
    monkeypatch.setattr(llm_module.litellm, "completion", spy)
    c = LLMClient("x", max_retries=0)
    c.complete(messages=[{"role": "user", "content": "hi"}])
    assert "tools" not in captured
    assert "tool_choice" not in captured
