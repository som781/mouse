"""Tests for the skill matcher."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from mouse.skills import match_skills, render_skills_for_prompt
from mouse.skills.loader import SkillRegistry, discover_skills
from mouse.skills.parser import parse_skill_file


def write_skill(root: Path, name: str, body: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(dedent(body).lstrip("\n"))
    return p


@pytest.fixture
def registry(tmp_path: Path) -> SkillRegistry:
    write_skill(tmp_path, "testing", """
        ---
        name: testing
        description: Write Python tests.
        triggers:
          - keywords: [test, pytest, coverage]
          - file_pattern: "test_*.py"
        ---
        # Testing body
    """)
    write_skill(tmp_path, "git-workflow", """
        ---
        name: git-workflow
        description: Git operations.
        triggers:
          - keywords: [git, commit, "pull request", rebase]
        ---
        # Git body
    """)
    write_skill(tmp_path, "docs", """
        ---
        name: docs
        description: Write docs.
        triggers:
          - keywords: [document, readme]
          - file_pattern: "README*"
        ---
        # Docs body
    """)
    return discover_skills(include_defaults=False, extra_dirs=[("test", tmp_path)])


def test_match_by_single_keyword(registry):
    hits = match_skills("can you help me write a test for this", registry)
    assert [s.name for s in hits] == ["testing"]


def test_match_is_case_insensitive(registry):
    hits = match_skills("GIT status please", registry)
    assert "git-workflow" in [s.name for s in hits]


def test_match_uses_word_boundary(registry):
    # "contest" must NOT match the "test" keyword.
    hits = match_skills("there is a contest today", registry)
    assert hits == []


def test_match_multi_word_keyword(registry):
    hits = match_skills("open a pull request on my behalf", registry)
    assert [s.name for s in hits] == ["git-workflow"]


def test_match_by_file_pattern(registry):
    hits = match_skills("please update test_utils.py", registry)
    assert [s.name for s in hits] == ["testing"]


def test_match_by_readme_pattern(registry):
    hits = match_skills("edit README.md with the new install steps", registry)
    names = [s.name for s in hits]
    assert "docs" in names


def test_match_multiple_skills(registry):
    hits = match_skills("commit the new test_foo.py file", registry)
    names = {s.name for s in hits}
    assert names == {"testing", "git-workflow"}


def test_empty_text_returns_nothing(registry):
    assert match_skills("", registry) == []


def test_empty_registry_returns_nothing():
    assert match_skills("test pytest commit", SkillRegistry()) == []


def test_no_match_returns_empty(registry):
    assert match_skills("weather is nice today", registry) == []


def test_render_empty_list():
    assert render_skills_for_prompt([]) == ""


def test_render_includes_body(tmp_path, registry):
    hits = match_skills("test this function", registry)
    rendered = render_skills_for_prompt(hits)
    assert "## Activated Skills" in rendered
    assert "### testing" in rendered
    assert "Testing body" in rendered
