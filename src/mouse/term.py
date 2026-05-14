"""ANSI terminal helpers."""

from __future__ import annotations

import os


class C:
    """ANSI color helpers for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GRAY    = "\033[90m"

    @staticmethod
    def styled(text: str, *styles: str) -> str:
        return "".join(styles) + text + C.RESET


def term_width(default: int = 80) -> int:
    """Get terminal width safely (works in IDEs, pipes, subprocesses)."""
    try:
        return min(os.get_terminal_size().columns, default)
    except OSError:
        return default
