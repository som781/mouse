"""Structured logging for Mouse.

A deliberately small logger — no handlers, no formatters, no hierarchy.
All output goes to stderr with a consistent prefix + color so it never
gets tangled up with model output on stdout.

Levels: DEBUG < INFO < WARN < ERROR. Set the minimum level with the
``$MOUSE_LOG`` env var (case-insensitive). Default is INFO.

Usage:
    from mouse.logging import get_logger
    log = get_logger("mcp.manager")
    log.info("connected to %s", name)
    log.error("connection failed: %s", err)

Percent-style formatting is used so expensive ``repr()`` calls can be
skipped when the level is disabled.
"""

from __future__ import annotations

import os
import sys
from enum import IntEnum

from mouse.term import C


class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


_NAME_TO_LEVEL = {
    "DEBUG": Level.DEBUG,
    "INFO": Level.INFO,
    "WARN": Level.WARN,
    "WARNING": Level.WARN,
    "ERROR": Level.ERROR,
}


def _current_level() -> Level:
    raw = os.environ.get("MOUSE_LOG", "INFO").strip().upper()
    return _NAME_TO_LEVEL.get(raw, Level.INFO)


_LEVEL_STYLES: dict[Level, tuple[str, str]] = {
    # (label, color)
    Level.DEBUG: ("debug", C.GRAY),
    Level.INFO:  ("info ", C.CYAN),
    Level.WARN:  ("warn ", C.YELLOW),
    Level.ERROR: ("error", C.RED),
}


class Logger:
    """A named logger. Cheap to create; share per module."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _log(self, level: Level, msg: str, args: tuple) -> None:
        if level < _current_level():
            return
        if args:
            try:
                msg = msg % args
            except Exception:
                # Never let a formatting bug crash the program — just
                # show the raw msg with the args appended.
                msg = f"{msg} {args!r}"
        label, color = _LEVEL_STYLES[level]
        prefix = C.styled(f"[{label}]", color, C.BOLD)
        name = C.styled(self.name, C.DIM)
        print(f"  {prefix} {name} {msg}", file=sys.stderr)

    def debug(self, msg: str, *args) -> None: self._log(Level.DEBUG, msg, args)
    def info(self,  msg: str, *args) -> None: self._log(Level.INFO,  msg, args)
    def warn(self,  msg: str, *args) -> None: self._log(Level.WARN,  msg, args)
    def error(self, msg: str, *args) -> None: self._log(Level.ERROR, msg, args)


def get_logger(name: str) -> Logger:
    """Return a logger for the given dotted module name."""
    return Logger(name)
