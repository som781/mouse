"""Dynamic tool pruner — drop irrelevant tools from each LLM call.

Most long-lived sessions accumulate tools the current turn will never
touch (a Slack MCP server, a Linear MCP server, a GitHub MCP server).
Shipping all of them every turn wastes tokens and — per the well-known
Vercel AI SDK lesson — measurably hurts tool-selection accuracy.

The pruner keeps:

    1. All "core" tools (non-MCP built-ins) — these are cheap and
       universally useful (bash, read_file, write_file, etc.)
    2. Any tool in ``always_keep_names`` — an escape hatch for a
       project that wants to pin a specific MCP tool.
    3. Any tool in ``sticky`` — tools already invoked in the current
       turn must stay available so the model can call them again
       without the schema vanishing mid-loop.
    4. The top-scoring optional (MCP) tools based on keyword overlap
       between the user's message and the tool's name + description.

Scoring is deliberately dumb: tokenize, intersect, count. Smarter
scoring (embeddings, LLM routing) can slot in behind the same
interface without changing the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mouse.tools.registry import Tool, ToolRegistry


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")

# Common English filler that appears in almost every tool description
# and user turn. Without this filter a sentence like "read the config
# file" would score a match against any tool description containing
# "the", which is useless.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "are", "was", "were", "has", "have", "had", "can", "will", "would",
    "should", "could", "may", "might", "must", "shall", "into", "onto",
    "out", "off", "over", "under", "any", "all", "some", "one", "two",
    "new", "old", "use", "used", "using", "get", "got", "set", "let",
    "like", "just", "now", "only", "also", "but", "not", "yet", "you",
    "your", "their", "them", "they", "its", "it's", "i'm", "i've",
})


def _tokens(text: str) -> set[str]:
    """Lowercase alpha-numeric tokens ≥ 3 chars, with English stopwords
    dropped. Short and generic tokens are too noisy to score on."""
    return {
        t.lower() for t in _WORD_RE.findall(text)
        if len(t) >= 3 and t.lower() not in _STOPWORDS
    }


@dataclass
class PruneConfig:
    """Knobs for the tool pruner."""

    # Keep every tool whose name doesn't start with an MCP prefix.
    # MCP tools are registered as ``mcp_<server>_<tool>`` so this cleanly
    # partitions "shipped with mouse" vs "from external server".
    keep_core: bool = True

    # Tool names pinned as always-available regardless of score.
    always_keep_names: set[str] = field(default_factory=set)

    # Max number of *optional* (scored) tools to include per call.
    max_optional: int = 15

    # Minimum keyword-overlap score to include an optional tool.
    # 0 means "include all" (pruner becomes score-order cap only).
    min_score: int = 1

    # Group-level unlock: if any tool from an MCP server (``mcp_<server>_*``)
    # makes it past scoring or is sticky, include every other tool from
    # the same server too. MCP servers are natural clusters — listing
    # orgs, getting an org, and updating an org belong together, and
    # asking the model to work on one without the others leads to
    # hallucinated IDs. Bypasses ``max_optional`` deliberately.
    group_unlock: bool = True


def _is_core(tool: Tool) -> bool:
    return not tool.name.startswith("mcp_")


def _mcp_server_of(name: str) -> str | None:
    """Return the MCP server prefix of ``name``, or ``None`` for core tools.

    ``mcp_globex_task_list_tasks`` → ``"globex"``. Names are registered
    as ``mcp_<server>_<rest>`` so the first two underscore-separated
    tokens uniquely identify the server.
    """
    if not name.startswith("mcp_"):
        return None
    parts = name.split("_", 2)
    if len(parts) < 3:
        return None
    return parts[1]


def _score(tool: Tool, user_tokens: set[str]) -> int:
    """Score ``tool`` against the user's token set.

    Two-tier scoring so exact-token matches still dominate over loose
    substring matches:

    * exact token hit → +2 (e.g. "organization" in user, "organization"
      in tool description)
    * substring hit   → +1 (e.g. "org" in user matches "organization"
      in tool description)

    The substring tier is what rescues short abbreviations like
    ``org``/``organization`` or ``repo``/``repository`` that the set-
    intersection scorer would miss entirely.
    """
    if not user_tokens:
        return 0
    # Split ``mcp_slack_post_message`` into {mcp, slack, post, message}
    # so name hits count.
    name_text = tool.name.replace("_", " ")
    full_text = f"{name_text} {tool.description}"
    tool_tokens = _tokens(full_text)
    tool_text_lower = full_text.lower()

    score = 0
    for ut in user_tokens:
        if ut in tool_tokens:
            score += 2
        elif len(ut) >= 3 and ut in tool_text_lower:
            score += 1
    return score


class ToolPruner:
    """Selects a relevant subset of tools for a given user turn."""

    def __init__(self, config: PruneConfig | None = None) -> None:
        self.config = config or PruneConfig()

    def select(
        self,
        text: str,
        registry: ToolRegistry,
        sticky: set[str] | None = None,
        *,
        recent_context: str = "",
    ) -> list[dict]:
        """Return tool schemas to send to the LLM for this turn.

        ``text`` is the user message (optionally augmented with skill
        hints by the caller). ``sticky`` are tool names already used in
        the current turn — they're always kept so the model can reuse
        them without the schema disappearing mid-loop.

        ``recent_context`` is a free-form string of recent prior turns'
        text. Tokens from it contribute to scoring at *half weight* of
        the current turn — enough to rescue follow-up turns like "show
        me the second one" (which has no domain signal of its own) by
        inheriting topic vocabulary from the previous turn, but not so
        much that they overwhelm the actual current request. Tokens
        already present in ``text`` are not double-counted.
        """
        cfg = self.config
        sticky = sticky or set()
        primary_tokens = _tokens(text)
        recent_tokens = _tokens(recent_context) - primary_tokens

        kept: list[Tool] = []
        seen: set[str] = set()

        def add(tool: Tool) -> None:
            if tool.name in seen:
                return
            kept.append(tool)
            seen.add(tool.name)

        # Pass 1: guaranteed-keep tools.
        for tool in registry.all():
            if (cfg.keep_core and _is_core(tool)) \
               or tool.name in cfg.always_keep_names \
               or tool.name in sticky:
                add(tool)

        # Pass 2: score the remainder and keep the top ``max_optional``.
        scored: list[tuple[int, Tool]] = []
        for tool in registry.all():
            if tool.name in seen:
                continue
            s = _score(tool, primary_tokens)
            if recent_tokens:
                # Half weight for recent context: only meaningful when
                # the recent turn has a strong (≥2) match against the
                # tool. Integer division means a single weak hit
                # rounds to 0 and doesn't drag in noise.
                s += _score(tool, recent_tokens) // 2
            if s >= cfg.min_score:
                scored.append((s, tool))

        # Stable sort: high score first, ties preserve registry order.
        scored.sort(key=lambda pair: -pair[0])
        for _, tool in scored[: cfg.max_optional]:
            add(tool)

        # Pass 3: MCP group unlock. If any tool from an MCP server
        # has been kept so far, bring in the rest of that server's
        # tools too. Servers are natural toolkits — calling
        # organization_get_organization without
        # organization_list_organizations is how hallucinated IDs get
        # born. This bypasses max_optional on purpose.
        if cfg.group_unlock:
            active_servers = {
                _mcp_server_of(t.name) for t in kept
                if _mcp_server_of(t.name) is not None
            }
            if active_servers:
                for tool in registry.all():
                    if tool.name in seen:
                        continue
                    if _mcp_server_of(tool.name) in active_servers:
                        add(tool)

        return [t.to_openai_schema() for t in kept]

    def select_names(
        self,
        text: str,
        registry: ToolRegistry,
        sticky: set[str] | None = None,
        *,
        recent_context: str = "",
    ) -> list[str]:
        """Same as :meth:`select` but returns names instead of schemas.

        Useful for logging / the `/tools` command.
        """
        schemas = self.select(
            text, registry, sticky=sticky, recent_context=recent_context
        )
        return [s["function"]["name"] for s in schemas]
