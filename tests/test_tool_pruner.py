"""Tests for the dynamic tool pruner."""

from __future__ import annotations

import pytest

from mouse.tools.pruner import PruneConfig, ToolPruner, _tokens
from mouse.tools.registry import PermissionLevel, Tool, ToolRegistry


def make_tool(name: str, description: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "ok",
        permission=PermissionLevel.SAFE,
    )


@pytest.fixture
def registry() -> ToolRegistry:
    """A registry with a few built-ins and several MCP tools."""
    r = ToolRegistry()
    # Core tools.
    r.register(make_tool("bash", "Run a shell command."))
    r.register(make_tool("read_file", "Read a file's contents."))
    r.register(make_tool("write_file", "Write a file."))
    # MCP tools covering several domains.
    r.register(make_tool("mcp_slack_post_message", "Send a message to a Slack channel."))
    r.register(make_tool("mcp_slack_list_channels", "List Slack channels the user can access."))
    r.register(make_tool("mcp_linear_create_issue", "Create a Linear issue in a project."))
    r.register(make_tool("mcp_linear_search_issues", "Search Linear issues by query."))
    r.register(make_tool("mcp_github_create_pr", "Open a GitHub pull request."))
    r.register(make_tool("mcp_weather_forecast", "Fetch the weather forecast for a location."))
    return r


def _names(schemas: list[dict]) -> list[str]:
    return [s["function"]["name"] for s in schemas]


# ─── Core keep behavior ──────────────────────────────────────────────


def test_core_tools_always_kept(registry):
    pruner = ToolPruner()
    kept = _names(pruner.select("anything unrelated xyzzy", registry))
    # All three core tools must be present even when nothing matches.
    assert {"bash", "read_file", "write_file"}.issubset(set(kept))


def test_unrelated_turn_drops_all_mcp(registry):
    pruner = ToolPruner()
    kept = _names(pruner.select("xyzzy plugh foobar", registry))
    assert not any(n.startswith("mcp_") for n in kept)


# ─── Keyword scoring ─────────────────────────────────────────────────


def test_slack_keyword_pulls_slack_tools(registry):
    pruner = ToolPruner()
    kept = _names(pruner.select("send a slack message to #eng", registry))
    assert "mcp_slack_post_message" in kept
    # The list_channels tool also matches on "slack" so should be included.
    assert "mcp_slack_list_channels" in kept
    # Unrelated MCP tools should be dropped.
    assert "mcp_weather_forecast" not in kept
    assert "mcp_linear_create_issue" not in kept


def test_multiple_domains_get_merged(registry):
    pruner = ToolPruner()
    kept = _names(pruner.select("open a pull request and file a linear issue", registry))
    assert "mcp_github_create_pr" in kept
    assert "mcp_linear_create_issue" in kept


def test_description_words_count(registry):
    pruner = ToolPruner()
    # "forecast" only appears in the description, not the tool name.
    kept = _names(pruner.select("what's the forecast tomorrow", registry))
    assert "mcp_weather_forecast" in kept


def test_substring_match_finds_morphological_variants(registry):
    """Short user tokens like 'org' must match longer tool tokens like
    'organization' via substring. This is the failure mode that broke
    a real globex session: 'globex org please' dropped the
    organization_list_organizations tool because 'org' ≠ 'organization'."""
    r = ToolRegistry()
    r.register(make_tool(
        "mcp_globex_organization_list_organizations",
        "List all organizations the user belongs to.",
    ))
    r.register(make_tool(
        "mcp_globex_task_list_tasks",
        "List tasks; filter by org_id, assignee, status.",
    ))
    pruner = ToolPruner()
    kept = _names(pruner.select("list my org please", r))
    assert "mcp_globex_organization_list_organizations" in kept


# ─── Recent context (follow-up turns) ────────────────────────────────


def test_recent_context_rescues_followup_turn_with_no_signal(registry):
    """A bare follow-up turn like 'show me the second one' has no
    domain vocabulary on its own. The pruner should still surface
    the topical tools by inheriting tokens from the prior turn via
    half-weight ``recent_context``."""
    pruner = ToolPruner(PruneConfig(min_score=2))
    # Current turn alone scores 0 against everything.
    kept_no_ctx = _names(pruner.select("show me the second one", registry))
    assert "mcp_slack_post_message" not in kept_no_ctx
    assert "mcp_slack_list_channels" not in kept_no_ctx
    # With prior-turn context that mentions slack, slack tools come back.
    kept_with_ctx = _names(pruner.select(
        "show me the second one",
        registry,
        recent_context="list slack channels in the eng workspace",
    ))
    assert "mcp_slack_list_channels" in kept_with_ctx


def test_recent_context_does_not_double_count_tokens(registry):
    """A token already in the current turn shouldn't get an extra
    half-weight bump just because it also appears in recent_context.
    The current turn's score should be unchanged by repetition."""
    pruner = ToolPruner()
    base = pruner.select_names("send a slack message", registry)
    duped = pruner.select_names(
        "send a slack message",
        registry,
        recent_context="slack slack slack",
    )
    assert set(base) == set(duped)


def test_recent_context_is_lower_weight_than_current_turn(registry):
    """Recent context contributes at half weight, so a strong current-
    turn match must outrank a strong recent-context match for the same
    score budget."""
    pruner = ToolPruner(PruneConfig(max_optional=1, min_score=1))
    kept = _names(pruner.select(
        "open a github pull request",
        registry,
        recent_context="search linear issues for the auth bug",
    ))
    # The single optional slot must go to github (current turn), not
    # linear (recent context).
    assert "mcp_github_create_pr" in kept


# ─── Sticky tools ────────────────────────────────────────────────────


def test_sticky_tool_stays_even_without_match(registry):
    pruner = ToolPruner()
    kept = _names(pruner.select(
        "unrelated text",
        registry,
        sticky={"mcp_weather_forecast"},
    ))
    assert "mcp_weather_forecast" in kept


# ─── Config knobs ────────────────────────────────────────────────────


def test_always_keep_names(registry):
    cfg = PruneConfig(always_keep_names={"mcp_github_create_pr"})
    pruner = ToolPruner(cfg)
    kept = _names(pruner.select("nothing relevant", registry))
    assert "mcp_github_create_pr" in kept


def test_max_optional_caps_results(registry):
    # group_unlock disabled so the cap isn't expanded by sibling-pull.
    cfg = PruneConfig(max_optional=1, group_unlock=False)
    pruner = ToolPruner(cfg)
    kept = _names(pruner.select("slack linear github weather", registry))
    # Core tools + at most 1 optional
    mcp_kept = [n for n in kept if n.startswith("mcp_")]
    assert len(mcp_kept) == 1


# ─── MCP group unlock ────────────────────────────────────────────────


def test_group_unlock_pulls_sibling_mcp_tools(registry):
    """When one globex/slack/etc. tool is kept, every tool from the
    same MCP server should come along for the ride — MCP servers are
    natural toolkits and splitting them causes hallucinated IDs."""
    # Tight max_optional so scoring alone can't pull in both.
    cfg = PruneConfig(max_optional=1, group_unlock=True)
    pruner = ToolPruner(cfg)
    # "post" only matches mcp_slack_post_message's name/description,
    # not list_channels, so without group_unlock list_channels would
    # be pruned.
    kept = _names(pruner.select("post to slack", registry))
    assert "mcp_slack_post_message" in kept
    assert "mcp_slack_list_channels" in kept
    # Other servers are still pruned.
    assert "mcp_linear_create_issue" not in kept


def test_group_unlock_from_sticky_only(registry):
    """Even a sticky tool with zero keyword score should unlock its
    server's siblings — this is the cross-turn MCP recovery case."""
    cfg = PruneConfig(group_unlock=True, min_score=1)
    pruner = ToolPruner(cfg)
    kept = _names(pruner.select(
        "unrelated text",
        registry,
        sticky={"mcp_globex_fake"},  # not in registry — ignored
    ))
    # Sticky not in registry is a no-op, check the real case:
    kept = _names(pruner.select(
        "unrelated text",
        registry,
        sticky={"mcp_slack_post_message"},
    ))
    assert "mcp_slack_post_message" in kept
    assert "mcp_slack_list_channels" in kept  # unlocked via group


def test_group_unlock_disabled_leaves_siblings_pruned(registry):
    cfg = PruneConfig(max_optional=1, group_unlock=False)
    pruner = ToolPruner(cfg)
    kept = _names(pruner.select("post to slack", registry))
    mcp_kept = [n for n in kept if n.startswith("mcp_")]
    assert len(mcp_kept) == 1  # only one slack tool, no group expansion


def test_min_score_zero_keeps_everything(registry):
    cfg = PruneConfig(min_score=0, max_optional=100)
    pruner = ToolPruner(cfg)
    kept = _names(pruner.select("xyzzy", registry))
    # With min_score=0 the pruner becomes pass-through.
    assert len(kept) == len(registry.list_names())


def test_keep_core_false_prunes_builtins(registry):
    cfg = PruneConfig(keep_core=False, min_score=1)
    pruner = ToolPruner(cfg)
    kept = _names(pruner.select("slack message", registry))
    # No keyword overlap means bash/read_file/write_file get dropped too.
    assert "bash" not in kept
    assert "mcp_slack_post_message" in kept


# ─── Tokenizer ───────────────────────────────────────────────────────


def test_tokens_ignores_short_words():
    assert "of" not in _tokens("a list of things")
    assert "list" in _tokens("a list of things")


def test_tokens_lowercases():
    assert _tokens("Slack Message") == {"slack", "message"}


# ─── select_names helper ─────────────────────────────────────────────


def test_select_names_returns_list(registry):
    pruner = ToolPruner()
    names = pruner.select_names("read the config file", registry)
    assert "read_file" in names
    assert isinstance(names, list)
