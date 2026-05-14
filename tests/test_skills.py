"""Tests for the skills parser and three-tier loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from mouse.errors import MouseError
from mouse.skills.loader import SkillRegistry, discover_skills
from mouse.skills.parser import Skill, SkillError, parse_skill_file


# ─── Helpers ─────────────────────────────────────────────────────────


def write_skill(root: Path, name: str, content: str) -> Path:
    """Create a SKILL.md under root/<name>/SKILL.md and return its path."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(dedent(content).lstrip("\n"))
    return path


VALID_SKILL = """
---
name: python-testing
description: Write and run Python tests using pytest.
triggers:
  - file_pattern: "test_*.py"
  - keywords: [test, pytest, coverage]
---

# Python Testing

Use pytest. Run with `pytest -q`.
"""


# ─── Parser ──────────────────────────────────────────────────────────


def test_skill_error_is_mouse_error():
    assert issubclass(SkillError, MouseError)


def test_parse_valid_skill(tmp_path: Path):
    p = write_skill(tmp_path, "python-testing", VALID_SKILL)
    skill = parse_skill_file(p)
    assert skill.name == "python-testing"
    assert "pytest" in skill.description
    assert skill.body.startswith("# Python Testing")
    assert skill.keywords == ["test", "pytest", "coverage"]
    assert skill.file_patterns == ["test_*.py"]
    assert skill.path == p


def test_parse_metadata_summary(tmp_path: Path):
    p = write_skill(tmp_path, "x", VALID_SKILL)
    skill = parse_skill_file(p)
    summary = skill.metadata_summary()
    assert summary.startswith("- python-testing:")


def test_parse_missing_top_delimiter(tmp_path: Path):
    p = write_skill(tmp_path, "x", "no frontmatter here\n")
    with pytest.raises(SkillError, match="missing '---' frontmatter delimiter"):
        parse_skill_file(p)


def test_parse_unclosed_frontmatter(tmp_path: Path):
    p = write_skill(tmp_path, "x", "---\nname: foo\ndescription: bar\nbody but no close\n")
    with pytest.raises(SkillError, match="not closed"):
        parse_skill_file(p)


def test_parse_missing_name(tmp_path: Path):
    p = write_skill(tmp_path, "x", "---\ndescription: just a description\n---\nbody\n")
    with pytest.raises(SkillError, match="missing required string 'name'"):
        parse_skill_file(p)


def test_parse_missing_description(tmp_path: Path):
    p = write_skill(tmp_path, "x", "---\nname: foo\n---\nbody\n")
    with pytest.raises(SkillError, match="missing required string 'description'"):
        parse_skill_file(p)


def test_parse_invalid_yaml(tmp_path: Path):
    p = write_skill(tmp_path, "x", "---\nname: foo\n  bad: : indent\n---\nbody\n")
    with pytest.raises(SkillError, match="invalid YAML"):
        parse_skill_file(p)


def test_parse_invalid_trigger_keywords_type(tmp_path: Path):
    p = write_skill(tmp_path, "x", """
        ---
        name: x
        description: x
        triggers:
          - keywords: "not a list"
        ---
        body
    """)
    with pytest.raises(SkillError, match="must be a list of strings"):
        parse_skill_file(p)


def test_parse_no_triggers_is_ok(tmp_path: Path):
    p = write_skill(tmp_path, "x", "---\nname: x\ndescription: y\n---\nbody\n")
    skill = parse_skill_file(p)
    assert skill.keywords == []
    assert skill.file_patterns == []


def test_parse_unknown_trigger_key_is_ignored(tmp_path: Path):
    """Forward-compat: a future trigger type shouldn't break loading."""
    p = write_skill(tmp_path, "x", """
        ---
        name: x
        description: y
        triggers:
          - keywords: [a]
          - some_future_thing: yes
        ---
        body
    """)
    skill = parse_skill_file(p)
    assert skill.keywords == ["a"]


# ─── Registry ────────────────────────────────────────────────────────


def test_registry_register_get_all(tmp_path: Path):
    r = SkillRegistry()
    p = write_skill(tmp_path, "a", VALID_SKILL)
    skill = parse_skill_file(p)
    r.register(skill)
    assert r.get("python-testing") is skill
    assert r.get("missing") is None
    assert len(r) == 1
    assert "python-testing" in r
    assert r.all() == [skill]


def test_registry_overwrite_keeps_latest(tmp_path: Path):
    r = SkillRegistry()
    p1 = write_skill(tmp_path / "first", "x", "---\nname: dup\ndescription: first\n---\nbody1\n")
    p2 = write_skill(tmp_path / "second", "x", "---\nname: dup\ndescription: second\n---\nbody2\n")
    r.register(parse_skill_file(p1))
    r.register(parse_skill_file(p2))
    assert r.get("dup").description == "second"


# ─── Three-tier discovery ────────────────────────────────────────────


def test_discover_with_extra_dirs(tmp_path: Path):
    builtin = tmp_path / "builtin"
    project = tmp_path / "project"
    user = tmp_path / "user"

    write_skill(builtin, "code-review", "---\nname: code-review\ndescription: builtin version\n---\nb\n")
    write_skill(project, "deploy",      "---\nname: deploy\ndescription: project version\n---\np\n")
    write_skill(user,    "personal",    "---\nname: personal\ndescription: user version\n---\nu\n")

    reg = discover_skills(
        include_defaults=False,
        extra_dirs=[
            ("builtin-test", builtin),
            ("project-test", project),
            ("user-test", user),
        ],
    )

    assert {s.name for s in reg.all()} >= {"code-review", "deploy", "personal"}
    assert reg.get("code-review").tier == "builtin-test"
    assert reg.get("deploy").tier == "project-test"
    assert reg.get("personal").tier == "user-test"


def test_discover_user_overrides_project_overrides_builtin(tmp_path: Path):
    builtin = tmp_path / "builtin"
    project = tmp_path / "project"
    user = tmp_path / "user"

    write_skill(builtin, "shared", "---\nname: shared\ndescription: from builtin\n---\nb\n")
    write_skill(project, "shared", "---\nname: shared\ndescription: from project\n---\np\n")
    write_skill(user,    "shared", "---\nname: shared\ndescription: from user\n---\nu\n")

    reg = discover_skills(
        include_defaults=False,
        extra_dirs=[
            ("builtin-test", builtin),
            ("project-test", project),
            ("user-test", user),
        ],
    )

    shared = reg.get("shared")
    assert shared.description == "from user"
    assert shared.tier == "user-test"


def test_discover_skips_malformed_files(tmp_path: Path, capsys: pytest.CaptureFixture):
    root = tmp_path / "skills"
    write_skill(root, "good", VALID_SKILL)
    write_skill(root, "bad",  "no frontmatter, just text\n")

    reg = discover_skills(include_defaults=False, extra_dirs=[("test", root)])

    assert reg.get("python-testing") is not None
    assert len(reg) == 1  # only the good one


def test_discover_missing_dir_is_silent(tmp_path: Path):
    reg = discover_skills(
        include_defaults=False,
        extra_dirs=[("nonexistent", tmp_path / "does-not-exist")],
    )
    assert isinstance(reg, SkillRegistry)
    assert len(reg) == 0
