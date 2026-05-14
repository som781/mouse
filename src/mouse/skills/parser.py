"""Parse SKILL.md files into Skill objects.

A skill file has YAML frontmatter and a markdown body, separated by ``---``::

    ---
    name: python-testing
    description: >
      Write and run pytest tests. Use when creating tests or fixing failures.
    triggers:
      - file_pattern: "test_*.py"
      - file_pattern: "*_test.py"
      - keywords: [test, pytest, coverage]
    ---

    # Python Testing

    ... markdown body ...

The body is preserved verbatim (minus surrounding whitespace) so it can
be injected into the model's context on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mouse.errors import MouseError


class SkillError(MouseError):
    """Raised when a SKILL.md file cannot be parsed."""


@dataclass
class Skill:
    """A parsed skill — metadata + body, with origin tracking."""

    name: str
    description: str
    body: str
    path: Path
    tier: str = "unknown"        # "builtin", "project", "user" — set by loader
    keywords: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)

    def metadata_summary(self) -> str:
        """One-line summary used in the system prompt's skill index."""
        return f"- {self.name}: {self.description}"


def parse_skill_file(path: Path) -> Skill:
    """Parse a SKILL.md file. Raises SkillError on malformed input."""
    try:
        text = path.read_text()
    except OSError as e:
        raise SkillError(f"{path}: cannot read file — {e}") from e

    frontmatter, body = _split_frontmatter(text, path)

    try:
        meta = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"{path}: invalid YAML in frontmatter — {e}") from e

    if not isinstance(meta, dict):
        raise SkillError(f"{path}: frontmatter must be a YAML mapping")

    name = meta.get("name")
    if not name or not isinstance(name, str):
        raise SkillError(f"{path}: frontmatter is missing required string 'name'")

    description = meta.get("description")
    if not description or not isinstance(description, str):
        raise SkillError(f"{path}: frontmatter is missing required string 'description'")

    keywords, file_patterns = _extract_triggers(meta.get("triggers"), path)

    return Skill(
        name=name.strip(),
        description=description.strip(),
        body=body.strip(),
        path=path,
        keywords=keywords,
        file_patterns=file_patterns,
    )


# ─── helpers ────────────────────────────────────────────────────────


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    """Return (frontmatter_yaml, body_markdown).

    The opening ``---`` must be the very first line. We split on the next
    ``---`` line to find the end of the frontmatter so values containing
    ``---`` (e.g. inside a multi-line string) don't trip the parser.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{path}: missing '---' frontmatter delimiter at top of file")

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        raise SkillError(f"{path}: frontmatter is not closed by a second '---'")

    frontmatter = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1:])
    return frontmatter, body


def _extract_triggers(raw: Any, path: Path) -> tuple[list[str], list[str]]:
    """Pull keyword and file_pattern lists out of the triggers block.

    The triggers block is a list of single-key mappings::

        triggers:
          - file_pattern: "*.py"
          - keywords: [test, pytest]

    Unknown trigger keys are ignored (forward-compatible). A missing
    triggers block is fine — the skill is then only loadable by name.
    """
    keywords: list[str] = []
    file_patterns: list[str] = []

    if raw is None:
        return keywords, file_patterns
    if not isinstance(raw, list):
        raise SkillError(f"{path}: 'triggers' must be a list of mappings")

    for entry in raw:
        if not isinstance(entry, dict):
            raise SkillError(f"{path}: each trigger entry must be a mapping, got {type(entry).__name__}")
        if "keywords" in entry:
            kws = entry["keywords"]
            if not isinstance(kws, list) or not all(isinstance(k, str) for k in kws):
                raise SkillError(f"{path}: 'keywords' must be a list of strings")
            keywords.extend(k.strip().lower() for k in kws if k.strip())
        if "file_pattern" in entry:
            pat = entry["file_pattern"]
            if not isinstance(pat, str):
                raise SkillError(f"{path}: 'file_pattern' must be a string")
            file_patterns.append(pat)

    return keywords, file_patterns
