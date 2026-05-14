"""Three-tier skill discovery.

Skills are discovered from three locations, with later tiers overriding
earlier ones (so a user can shadow a project skill, and a project can
shadow a built-in):

    1. builtin  — src/mouse/skills/built_in/<name>/SKILL.md
    2. project  — <project_root>/.harness/skills/<name>/SKILL.md
    3. user     — ~/.mouse/skills/<name>/SKILL.md

Each skill lives in its own directory so it can ship supporting files
(scripts, prompts, fixtures) alongside SKILL.md. We only parse SKILL.md;
the rest of the directory is the skill author's to organize.
"""

from __future__ import annotations

from pathlib import Path

from mouse.logging import get_logger
from mouse.project import PROJECT_ROOT
from mouse.skills.parser import Skill, SkillError, parse_skill_file

log = get_logger("mouse.skills")


# ─── Tier paths ──────────────────────────────────────────────────────


def builtin_skills_dir() -> Path:
    """Built-in skills shipped with the package (src/mouse/skills/built_in/)."""
    return Path(__file__).resolve().parent / "built_in"


def project_skills_dir() -> Path:
    """Project-level skills checked into the repo."""
    return Path(PROJECT_ROOT) / ".harness" / "skills"


def user_skills_dir() -> Path:
    """User-level personal skills."""
    return Path.home() / ".mouse" / "skills"


# ─── Registry ────────────────────────────────────────────────────────


class SkillRegistry:
    """In-memory index of all discovered skills, keyed by name.

    The registry only holds metadata + body; rendering the skill into a
    prompt is the caller's job. We track which tier each skill came from
    so callers (and the /skills command) can show the source.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Add or replace a skill. Later registrations override earlier ones,
        which is how user > project > builtin precedence works."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        """All skills in registration order."""
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills


# ─── Discovery ───────────────────────────────────────────────────────


_TIERS: list[tuple[str, callable]] = [
    ("builtin", builtin_skills_dir),
    ("project", project_skills_dir),
    ("user",    user_skills_dir),
]


def discover_skills(
    *,
    extra_dirs: list[tuple[str, Path]] | None = None,
    include_defaults: bool = True,
) -> SkillRegistry:
    """Walk all three tiers and return a populated registry.

    ``extra_dirs`` is a hook for tests — pairs of (tier_label, path) that
    are scanned in addition to the default tiers, in the given order.
    ``include_defaults=False`` skips the built-in/project/user tiers,
    which tests use to isolate discovery to ``extra_dirs`` only.
    """
    registry = SkillRegistry()

    tiers: list[tuple[str, Path]] = []
    if include_defaults:
        tiers.extend((label, getter()) for label, getter in _TIERS)
    if extra_dirs:
        tiers.extend(extra_dirs)

    for tier, root in tiers:
        if not root.exists() or not root.is_dir():
            continue
        for skill_md in sorted(root.glob("*/SKILL.md")):
            try:
                skill = parse_skill_file(skill_md)
            except SkillError as e:
                log.warn("skipping malformed skill: %s", e)
                continue
            skill.tier = tier
            registry.register(skill)
            log.debug("loaded skill '%s' from %s tier (%s)", skill.name, tier, skill_md)

    return registry
