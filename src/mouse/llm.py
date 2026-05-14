"""LLMClient — thin wrapper around litellm.

Responsibilities:
  - Route completions through a single place so retries, auth,
    error wrapping, and token counting don't scatter across the codebase.
  - Provide a typed LLMError so callers can distinguish API failures
    from programming errors.
  - Let the agent loop stay focused on orchestration.

Streaming is intentionally not implemented yet — add it when something
actually consumes streamed tokens (UI, long generations).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import litellm

from mouse.errors import MouseError


def _is_retriable(exc: BaseException) -> bool:
    """Return True if it's worth retrying the same request after an error.

    Bad requests (including ContextWindowExceededError, schema errors,
    and most authentication failures) will not get better on retry —
    surface them immediately. Network blips, rate limits, and transient
    server errors are retriable.
    """
    # Try to use litellm's typed exceptions when available; fall back to
    # name matching so we don't crash if the SDK reorganizes its module.
    bad_request = getattr(litellm, "BadRequestError", None) or getattr(
        getattr(litellm, "exceptions", None), "BadRequestError", None
    )
    auth_error = getattr(litellm, "AuthenticationError", None) or getattr(
        getattr(litellm, "exceptions", None), "AuthenticationError", None
    )
    not_found = getattr(litellm, "NotFoundError", None) or getattr(
        getattr(litellm, "exceptions", None), "NotFoundError", None
    )
    non_retriable_types = tuple(t for t in (bad_request, auth_error, not_found) if t)
    if non_retriable_types and isinstance(exc, non_retriable_types):
        return False
    # Defensive: catch ContextWindowExceededError by name in case the
    # class hierarchy changes.
    if "ContextWindowExceeded" in type(exc).__name__:
        return False
    return True


# Suppress LiteLLM's noisy logs once, at import time.
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")


class LLMError(MouseError):
    """Raised when an LLM call fails after all retries are exhausted."""

    def __init__(self, message: str, *, attempts: int, last_exception: BaseException | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass
class Usage:
    """Token usage for a single completion. Zero on providers that don't report it."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls()
        return cls(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class LLMClient:
    """Thin, retrying wrapper around litellm.completion.

    The client is stateful only in the sense that it remembers the
    current model and retry policy — callers pass messages per-call.
    """

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def set_model(self, model: str) -> None:
        """Swap the active model. Used by /model slash command."""
        self.model = model

    def context_window(self) -> int:
        """Return the model's advertised context window, in tokens.

        Prefers LiteLLM's ``get_model_info`` lookup so the number tracks
        the model registry as new models ship. Falls back to a
        conservative 32k when the lookup can't resolve the model — that
        way the auto-compact trigger stays active rather than silently
        disabling itself on an unknown model id.
        """
        try:
            info = litellm.get_model_info(self.model)
            window = info.get("max_input_tokens") or info.get("max_tokens")
            if isinstance(window, int) and window > 0:
                return window
        except Exception:
            pass
        # Conservative fallback: enough that auto-compact still fires
        # on long sessions but small enough that an unknown 128k model
        # isn't treated as limitless.
        return 32_000

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int | None = None,
    ) -> Any:
        """Call the LLM with retries. Returns the raw litellm response object.

        Callers are responsible for extracting `.choices[0].message` etc. —
        the wrapper stays deliberately thin so it doesn't hide the SDK shape.
        """
        # Build kwargs, omitting None so litellm uses its own defaults.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_exc: BaseException | None = None
        attempts = 0
        # Total tries = max_retries + 1 (the initial attempt).
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                return litellm.completion(**kwargs)
            except Exception as e:  # noqa: BLE001 — litellm raises many types
                last_exc = e
                # Don't waste retries on errors that won't fix themselves.
                if not _is_retriable(e):
                    break
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds)
                    continue
                break

        raise LLMError(
            f"LLM call failed after {attempts} attempt(s): {last_exc}",
            attempts=attempts,
            last_exception=last_exc,
        )
