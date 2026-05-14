"""Skill matcher — decide which skills apply to a user turn.

The matcher is intentionally dumb: keyword and file-glob matching against
the user's message text. Smarter activation (semantic similarity, LLM
routing) can be layered on later without changing the call site.

Matching rules:
    - Keywords match as whole words, case-insensitive. Multi-word
      keywords ("pull request") are matched as substrings since regex
      word boundaries would reject the space.
    - File patterns are matched against tokens in the text using
      ``fnmatch`` (so ``*.py`` matches ``foo.py`` anywhere in the line).
    - A skill is activated if **any** trigger matches.
"""

from __future__ import annotations

import fnmatch
import re

from mouse.skills.loader import SkillRegistry
from mouse.skills.parser import Skill


_TOKEN_RE = re.compile(r"[\w./\-]+")


def _keyword_matches(keyword: str, text_lower: str) -> bool:
    kw = keyword.lower().strip()
    if not kw:
        return False
    if " " in kw:
        # Multi-word keywords: plain substring match.
        return kw in text_lower
    return re.search(rf"\b{re.escape(kw)}\b", text_lower) is not None


def _pattern_matches(pattern: str, tokens: list[str]) -> bool:
    return any(fnmatch.fnmatch(tok, pattern) for tok in tokens)


def match_skills(text: str, registry: SkillRegistry) -> list[Skill]:
    """Return skills whose triggers fire for ``text``.

    Order is stable (insertion order of the registry, which is
    builtin → project → user after a default discover call).
    """
    if not text or len(registry) == 0:
        return []

    text_lower = text.lower()
    tokens = _TOKEN_RE.findall(text)

    matched: list[Skill] = []
    for skill in registry.all():
        fired = any(_keyword_matches(kw, text_lower) for kw in skill.keywords) \
            or any(_pattern_matches(pat, tokens) for pat in skill.file_patterns)
        if fired:
            matched.append(skill)
    return matched


def render_skills_for_prompt(skills: list[Skill]) -> str:
    """Format activated skills as a block to inject into the system prompt.

    Returns an empty string if no skills are active so callers can join
    unconditionally.
    """
    if not skills:
        return ""
    parts = ["## Activated Skills",
             "The following skill guides apply to the current task. "
             "Follow their guidance when relevant.\n"]
    for skill in skills:
        parts.append(f"### {skill.name}\n{skill.body.strip()}\n")
    return "\n".join(parts)
