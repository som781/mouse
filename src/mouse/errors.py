"""Error hierarchy for Mouse.

All Mouse-raised exceptions should inherit from :class:`MouseError`. This
lets top-level handlers (the CLI, SDK entry points, tests) catch every
expected failure with a single except clause while still letting
programming bugs (``TypeError``, ``AttributeError``) propagate normally.

New error classes should only be added when callers actually need to
distinguish them. When in doubt, raise ``MouseError`` directly.
"""

from __future__ import annotations


class MouseError(Exception):
    """Base class for all Mouse exceptions."""


# Concrete subclasses live next to the code that raises them
# (ConfigError in mouse.config, LLMError in mouse.llm) and inherit from
# this base. They are re-exported here for convenience.

def __getattr__(name: str):
    # Lazy re-export to avoid import cycles (errors.py is imported by
    # logging.py which is imported very early).
    if name == "ConfigError":
        from mouse.config import ConfigError
        return ConfigError
    if name == "LLMError":
        from mouse.llm import LLMError
        return LLMError
    raise AttributeError(name)


__all__ = ["MouseError", "ConfigError", "LLMError"]
