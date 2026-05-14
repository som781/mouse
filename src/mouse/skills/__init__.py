"""Skills engine — progressive disclosure of behavior packages.

A skill is a SKILL.md file with YAML frontmatter (name, description,
triggers) plus a markdown body. At startup we load only the metadata so
the system prompt stays lean. The body is loaded on-demand when the
skill matches the current task and evicted afterwards.
"""

from mouse.skills.parser import Skill, SkillError, parse_skill_file
from mouse.skills.loader import (
    SkillRegistry,
    discover_skills,
    builtin_skills_dir,
    project_skills_dir,
    user_skills_dir,
)
from mouse.skills.matcher import match_skills, render_skills_for_prompt

__all__ = [
    "Skill",
    "SkillError",
    "parse_skill_file",
    "SkillRegistry",
    "discover_skills",
    "builtin_skills_dir",
    "project_skills_dir",
    "user_skills_dir",
    "match_skills",
    "render_skills_for_prompt",
]
