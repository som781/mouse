"""Project root detection and context loading."""

from __future__ import annotations

import os
from pathlib import Path


def detect_project_root() -> str:
    """
    Walk up from cwd to find the project root (looks for .git, package.json, etc.).
    Falls back to cwd if nothing found.
    """
    markers = {".git", "package.json", "pyproject.toml", "Cargo.toml",
               "go.mod", "pom.xml", "Makefile", "setup.py", ".hg"}
    current = Path(os.getcwd()).resolve()
    for directory in [current, *current.parents]:
        if any((directory / m).exists() for m in markers):
            return str(directory)
        if directory == directory.parent:
            break
    return str(current)


# Detect once at import
PROJECT_ROOT = detect_project_root()


def load_project_context() -> str:
    """
    Auto-detect project context by reading AGENTS.md, README, package.json, etc.
    This is the 'feedforward guide' — giving the agent context before it acts.
    """
    context_parts = []
    root = Path(PROJECT_ROOT)

    # Check for agent-specific instructions
    for agent_file in ["AGENTS.md", "CLAUDE.md", ".cursorrules", "COPILOT.md"]:
        p = root / agent_file
        if p.exists():
            content = p.read_text()[:2000]
            context_parts.append(f"### {agent_file}\n{content}")

    # Check for project metadata
    for meta_file in ["package.json", "pyproject.toml", "Cargo.toml", "go.mod"]:
        p = root / meta_file
        if p.exists():
            content = p.read_text()[:1000]
            context_parts.append(f"### {meta_file}\n{content}")

    # Check for README
    for readme in ["README.md", "readme.md", "README.rst", "README.txt"]:
        p = root / readme
        if p.exists():
            content = p.read_text()[:1500]
            context_parts.append(f"### {readme}\n{content}")

    if context_parts:
        return "\n\n".join(context_parts)

    return f"No project files detected. Project root: {PROJECT_ROOT}"
