"""Tests for the Agent loop using a scripted FakeLLMClient.

These tests don't touch the network — the FakeLLMClient injected via the
`llm` constructor argument returns canned FakeResponses, so we can verify
the loop's plumbing (tool dispatch, history append, stats, termination).
"""

from __future__ import annotations

import json

import pytest

from mouse.engine.agent import (
    Agent,
    AgentConfig,
    AgentStats,
    MAX_TOOL_RESULT_CHARS,
    _truncate_tool_result,
)
from mouse.llm import Usage
from mouse.tools.permissions import PermissionManager
from mouse.tools.registry import PermissionLevel, Tool, ToolRegistry

from tests.conftest import FakeFunction, FakeMessage, FakeResponse, FakeToolCall, FakeUsage


# ─── Helpers ─────────────────────────────────────────────────────────


def make_agent(llm, registry: ToolRegistry | None = None, auto_approve: bool = True) -> Agent:
    return Agent(
        config=AgentConfig(model="fake-model", max_steps=5),
        tools=registry or ToolRegistry(),
        permissions=PermissionManager(auto_approve=auto_approve),
        llm=llm,
    )


def tool_call(name: str, args: dict, tc_id: str = "tc1") -> FakeToolCall:
    return FakeToolCall(id=tc_id, function=FakeFunction(name=name, arguments=json.dumps(args)))


# ─── AgentStats ──────────────────────────────────────────────────────


def test_agent_stats_record_usage():
    s = AgentStats()
    s.record(Usage(input_tokens=10, output_tokens=5))
    s.record(Usage(input_tokens=2, output_tokens=3))
    assert s.total_input_tokens == 12
    assert s.total_output_tokens == 8
    summary = s.summary()
    assert "20" in summary  # 12 + 8 total


# ─── Agent loop ──────────────────────────────────────────────────────


def test_terminates_on_text_only_response(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content="all done")))
    agent = make_agent(llm)

    out = agent.chat("hi")

    assert out == "all done"
    assert agent.stats.total_steps == 1
    assert agent.stats.total_tool_calls == 0
    # The user message + assistant reply should be in history.
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.messages[-1]["content"] == "all done"


def test_dispatches_tool_call_then_finishes(make_llm):
    # Build a registry with a single, deterministic tool.
    registry = ToolRegistry()
    seen_args: dict = {}
    def echo(args):
        seen_args.update(args)
        return f"echoed:{args.get('msg')}"
    registry.register(Tool(
        name="echo", description="", parameters={}, handler=echo,
        permission=PermissionLevel.SAFE,
    ))

    # Step 1: model calls echo. Step 2: model returns text and stops.
    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[tool_call("echo", {"msg": "hello"})])),
        FakeResponse(FakeMessage(content="finished")),
    )
    agent = make_agent(llm, registry=registry)

    result = agent.chat("please echo hello")

    assert result == "finished"
    assert seen_args == {"msg": "hello"}
    assert agent.stats.total_tool_calls == 1
    assert agent.stats.total_steps == 2

    # The conversation history should contain a tool message with the
    # echoed result, keyed to the original tool_call_id.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "echoed:hello"
    assert tool_msgs[0]["tool_call_id"] == "tc1"


def test_unknown_tool_returns_error_to_model(make_llm):
    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[tool_call("nope", {})])),
        FakeResponse(FakeMessage(content="ok")),
    )
    agent = make_agent(llm, registry=ToolRegistry())

    agent.chat("call a fake tool")

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs[0]["content"].startswith("ERROR: Unknown tool")


def test_tool_handler_exception_is_caught(make_llm):
    registry = ToolRegistry()
    def boom(_args):
        raise RuntimeError("kaboom")
    registry.register(Tool(
        name="boom", description="", parameters={}, handler=boom,
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[tool_call("boom", {})])),
        FakeResponse(FakeMessage(content="recovered")),
    )
    agent = make_agent(llm, registry=registry)

    agent.chat("trigger boom")

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert "kaboom" in tool_msgs[0]["content"]
    assert agent.stats.total_errors == 1


def test_permission_denied_returns_denied_message(make_llm):
    registry = ToolRegistry()
    called = {"n": 0}
    def handler(_args):
        called["n"] += 1
        return "should not run"
    registry.register(Tool(
        name="restricted", description="", parameters={}, handler=handler,
        permission=PermissionLevel.DENY,
    ))

    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[tool_call("restricted", {})])),
        FakeResponse(FakeMessage(content="ok")),
    )
    # auto_approve does NOT bypass DENY tools.
    agent = make_agent(llm, registry=registry, auto_approve=False)

    agent.chat("try restricted")

    assert called["n"] == 0
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert "DENIED" in tool_msgs[0]["content"]


def test_max_steps_respected(make_llm):
    # Model never stops calling tools — agent should bail at max_steps.
    registry = ToolRegistry()
    registry.register(Tool(
        name="loop", description="", parameters={},
        handler=lambda a: "again", permission=PermissionLevel.SAFE,
    ))
    # Provide a single response that always asks for another tool call.
    looping = FakeResponse(FakeMessage(tool_calls=[tool_call("loop", {})]))
    llm = make_llm(looping)  # FakeLLMClient returns the last response forever
    agent = make_agent(llm, registry=registry)
    agent.config.max_steps = 3

    agent.chat("go")

    assert agent.stats.total_steps == 3


def test_truncate_tool_result_short_unchanged():
    assert _truncate_tool_result("hello") == "hello"


def test_truncate_tool_result_caps_long_payload():
    payload = "x" * (MAX_TOOL_RESULT_CHARS * 3)
    out = _truncate_tool_result(payload)
    assert len(out) <= MAX_TOOL_RESULT_CHARS
    assert "truncated" in out


def test_huge_tool_output_is_capped_in_history(make_llm):
    """A pathological large tool response must not bloat agent.messages."""
    registry = ToolRegistry()
    registry.register(Tool(
        name="firehose", description="", parameters={},
        handler=lambda a: "X" * 1_000_000,  # 1 MB blob
        permission=PermissionLevel.SAFE,
    ))
    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[tool_call("firehose", {})])),
        FakeResponse(FakeMessage(content="ok")),
    )
    agent = make_agent(llm, registry=registry)

    agent.chat("call firehose")

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    stored = tool_msgs[0]["content"]
    assert len(stored) <= MAX_TOOL_RESULT_CHARS
    assert "truncated" in stored


def test_reset_clears_history_and_stats(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content="hi")))
    agent = make_agent(llm)
    agent.chat("hello")
    assert len(agent.messages) > 0
    agent.reset()
    assert agent.messages == []
    assert agent.stats.total_steps == 0


# ─── Sprint 2 wiring: skills, pruner, tool_output_store ──────────────


def test_skills_activate_and_inject_into_system_prompt(make_llm, tmp_path):
    """Matching skills should be attached to the agent and their bodies
    should appear in the per-turn system message."""
    from mouse.skills import discover_skills
    from tests.test_skills import write_skill

    write_skill(tmp_path, "git-workflow", """
        ---
        name: git-workflow
        description: Handle git operations.
        triggers:
          - keywords: [git, commit]
        ---
        # Git body marker XYZZY_UNIQUE
    """)
    skills = discover_skills(include_defaults=False, extra_dirs=[("test", tmp_path)])

    llm = make_llm(FakeResponse(FakeMessage(content="done")))
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=ToolRegistry(),
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        skills=skills,
    )

    agent.chat("can you commit this for me")

    # The matcher should have activated the skill for this turn.
    assert [s.name for s in agent.active_skills] == ["git-workflow"]
    # The scripted LLM recorded the messages it was called with — the
    # turn-local system prompt should contain the skill body marker.
    sent = llm.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "XYZZY_UNIQUE" in sent[0]["content"]


def test_skills_not_activated_when_text_does_not_match(make_llm, tmp_path):
    from mouse.skills import discover_skills
    from tests.test_skills import write_skill

    write_skill(tmp_path, "git-workflow", """
        ---
        name: git-workflow
        description: Handle git operations.
        triggers:
          - keywords: [git, commit]
        ---
        # Git body marker XYZZY_UNIQUE
    """)
    skills = discover_skills(include_defaults=False, extra_dirs=[("test", tmp_path)])

    llm = make_llm(FakeResponse(FakeMessage(content="done")))
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=ToolRegistry(),
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        skills=skills,
    )

    agent.chat("unrelated question about weather")

    assert agent.active_skills == []
    sent = llm.calls[0]["messages"]
    assert "XYZZY_UNIQUE" not in sent[0]["content"]


def test_pruner_limits_tools_sent_to_llm(make_llm):
    """Agent must forward the pruner's output to the LLM, not all tools."""
    from mouse.tools.pruner import PruneConfig, ToolPruner

    registry = ToolRegistry()
    registry.register(Tool(name="bash", description="shell", parameters={},
                           handler=lambda a: "", permission=PermissionLevel.SAFE))
    registry.register(Tool(name="mcp_slack_post", description="Send to Slack",
                           parameters={}, handler=lambda a: "",
                           permission=PermissionLevel.SAFE))
    registry.register(Tool(name="mcp_weather", description="Get weather forecast",
                           parameters={}, handler=lambda a: "",
                           permission=PermissionLevel.SAFE))

    llm = make_llm(FakeResponse(FakeMessage(content="done")))
    pruner = ToolPruner(PruneConfig(max_optional=5, min_score=1))
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        pruner=pruner,
    )

    agent.chat("post a slack message to the team")

    schemas = llm.calls[0]["tools"]
    names = {s["function"]["name"] for s in schemas}
    assert "bash" in names                   # core always kept
    assert "mcp_slack_post" in names         # matched
    assert "mcp_weather" not in names        # pruned


def test_pruner_keeps_tools_sticky_after_first_use(make_llm):
    """Once a tool is called in a turn it must stay visible across the
    remaining steps of that turn even if its keywords don't match."""
    from mouse.tools.pruner import PruneConfig, ToolPruner

    registry = ToolRegistry()
    registry.register(Tool(name="bash", description="shell", parameters={},
                           handler=lambda a: "", permission=PermissionLevel.SAFE))
    registry.register(Tool(
        name="mcp_linear_create_issue",
        description="Create a Linear issue",
        parameters={}, handler=lambda a: "issue created",
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("mcp_linear_create_issue", {"title": "x"})
        ])),
        FakeResponse(FakeMessage(content="ok")),
    )
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        pruner=ToolPruner(PruneConfig(max_optional=5, min_score=1)),
    )

    # User text that matches "linear" so the tool is in scope initially.
    agent.chat("file a linear issue for the flaky test")

    # Second call should still include the tool even though on step 2
    # no new user text is added — sticky set keeps it.
    step2_schemas = llm.calls[1]["tools"]
    step2_names = {s["function"]["name"] for s in step2_schemas}
    assert "mcp_linear_create_issue" in step2_names


def test_pruner_sticky_persists_across_user_turns(make_llm):
    """A tool used in turn 1 must stay exposed on turn 2 even if the
    second user message has zero keyword overlap. Without this, a
    multi-turn workflow can lose access to a tool it already used
    (turn 1 discovers something, turn 2 acts on it, but the action
    tool was re-pruned out on the second turn)."""
    from mouse.tools.pruner import PruneConfig, ToolPruner

    registry = ToolRegistry()
    registry.register(Tool(name="bash", description="shell", parameters={},
                           handler=lambda a: "", permission=PermissionLevel.SAFE))
    registry.register(Tool(
        name="svc_list_organizations",
        description="List all organizations the user belongs to.",
        parameters={}, handler=lambda a: '[{"id":"abc","name":"x"}]',
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        # Turn 1: model calls the org-list tool, then finishes.
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("svc_list_organizations", {})
        ])),
        FakeResponse(FakeMessage(content="done")),
        # Turn 2: just text — we're inspecting what tools were sent.
        FakeResponse(FakeMessage(content="ok")),
    )
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        pruner=ToolPruner(PruneConfig(max_optional=5, min_score=1)),
    )

    # Turn 1 mentions "organizations" so the tool is in scope.
    agent.chat("list my organizations")
    # Turn 2 has no overlapping keywords with the tool at all.
    agent.chat("fetch tasks on me")

    # The third LLM call (= turn 2, step 1) must still include the
    # org-list tool because it was sticky from turn 1.
    turn2_schemas = llm.calls[2]["tools"]
    turn2_names = {s["function"]["name"] for s in turn2_schemas}
    assert "svc_list_organizations" in turn2_names


def test_agent_reset_clears_sticky_tools(make_llm):
    """reset() must drop session-wide sticky state so a new session
    starts from a clean pruning baseline."""
    from mouse.tools.pruner import PruneConfig, ToolPruner

    registry = ToolRegistry()
    registry.register(Tool(name="bash", description="shell", parameters={},
                           handler=lambda a: "", permission=PermissionLevel.SAFE))
    registry.register(Tool(
        name="mcp_linear_create_issue",
        description="Create a Linear issue",
        parameters={}, handler=lambda a: "ok",
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("mcp_linear_create_issue", {"title": "x"})
        ])),
        FakeResponse(FakeMessage(content="done")),
        FakeResponse(FakeMessage(content="ok")),
    )
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        pruner=ToolPruner(PruneConfig(max_optional=5, min_score=1)),
    )

    agent.chat("file a linear issue for the flaky test")
    assert "mcp_linear_create_issue" in agent.sticky_tools
    agent.reset()
    assert agent.sticky_tools == set()


def test_agent_saves_full_tool_output_before_truncation(make_llm):
    """The full (pre-truncation) tool output must land in the store so
    fetch_tool_output can retrieve the omitted portion."""
    from mouse.tools import tool_output_store

    tool_output_store.clear()
    try:
        registry = ToolRegistry()
        full_blob = "LINE_A\n" + ("x" * 100_000) + "\nLINE_Z"
        registry.register(Tool(
            name="firehose", description="", parameters={},
            handler=lambda a: full_blob,
            permission=PermissionLevel.SAFE,
        ))

        llm = make_llm(
            FakeResponse(FakeMessage(tool_calls=[tool_call("firehose", {}, tc_id="call_99")])),
            FakeResponse(FakeMessage(content="ok")),
        )
        agent = make_agent(llm, registry=registry)
        agent.chat("fetch everything")

        # The tool message in history is truncated.
        tool_msg = [m for m in agent.messages if m.get("role") == "tool"][0]
        assert "truncated" in tool_msg["content"]
        assert len(tool_msg["content"]) <= MAX_TOOL_RESULT_CHARS

        # But the store keeps the full version keyed by tool_call_id.
        full = tool_output_store.fetch("call_99", offset=0, limit=200_000)
        assert "LINE_A" in full
        assert "LINE_Z" in full
    finally:
        tool_output_store.clear()


# ─── Grounding integration ──────────────────────────────────────────


def test_agent_refuses_dispatch_on_ungrounded_id_arg(make_llm):
    """When the model calls a tool with an id-shaped arg it never
    actually discovered, the harness must intercept the call and feed
    a synthetic refusal back instead of dispatching. The real tool
    handler should never run on the fabricated call."""
    calls: list[dict] = []

    registry = ToolRegistry()
    registry.register(Tool(
        name="svc_list_items", description="", parameters={},
        handler=lambda a: (calls.append(a) or '{"items":[]}'),
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        # Turn 1: model invents a child_id.
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("svc_list_items",
                      {"child_ids": ["fake-child"]},
                      tc_id="c1"),
        ])),
        # Turn 2: model recovers and finishes.
        FakeResponse(FakeMessage(content="understood")),
    )
    agent = make_agent(llm, registry=registry)
    agent.chat("show me the latest items")

    # The real handler was NEVER called with the fabricated arg.
    assert calls == []

    # The refusal landed in the tool message stream for the model.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "HARNESS REFUSED" in tool_msgs[0]["content"]
    assert "child_ids='fake-child'" in tool_msgs[0]["content"]


def test_agent_allows_dispatch_when_id_was_seen_in_prior_output(make_llm):
    """After a discovery call surfaces real ids, the next call that
    reuses one of them must dispatch normally."""
    dispatched: list[dict] = []

    registry = ToolRegistry()
    registry.register(Tool(
        name="svc_list_children", description="", parameters={},
        handler=lambda a: '[{"uuid":"real-child","name":"Alpha"}]',
        permission=PermissionLevel.SAFE,
    ))
    registry.register(Tool(
        name="svc_list_items", description="", parameters={},
        handler=lambda a: (dispatched.append(a) or '{"items":[{"id":"t1"}]}'),
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("svc_list_children", {}, tc_id="c1"),
        ])),
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("svc_list_items", {"child_ids": ["real-child"]}, tc_id="c2"),
        ])),
        FakeResponse(FakeMessage(content="done")),
    )
    agent = make_agent(llm, registry=registry)
    agent.chat("list the items under each child")

    # The second call dispatched because the id came from step 1.
    assert dispatched == [{"child_ids": ["real-child"]}]


def test_agent_reset_clears_grounding(make_llm):
    llm = make_llm(FakeResponse(FakeMessage(content="hi")))
    agent = make_agent(llm)
    agent.chat("remember uuid-AAA-BBB-CCC")
    assert agent.grounding.check_args({"user_id": "uuid-AAA-BBB-CCC"}) == []
    agent.reset()
    assert agent.grounding.check_args({"user_id": "uuid-AAA-BBB-CCC"}) != []


def test_replace_messages_resyncs_grounding_to_visible_content():
    """After replace_messages (e.g. after /compact), grounding must
    only approve ids the model can still see in its message context —
    otherwise the gate lets through fabricated ids that happen to
    substring-match content the model has lost access to."""
    registry = ToolRegistry()
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=None,  # not used in this test path
    )
    # Seed as if the agent had seen a pair of ids in a full session.
    agent.grounding.record_user_message("please use id abc-KEPT-123")
    agent.grounding.record_tool_result("svc", '{"id":"abc-DROPPED-999"}')
    assert agent.grounding.check_args({"user_id": "abc-KEPT-123"}) == []
    assert agent.grounding.check_args({"target_id": "abc-DROPPED-999"}) == []

    # Simulate a compaction that retained only the user message with
    # the KEPT id and dropped the tool result with the DROPPED id.
    agent.replace_messages([
        {"role": "user", "content": "please use id abc-KEPT-123"},
        {"role": "assistant", "content": "acknowledged"},
    ])

    # KEPT id is still visible to the model — grounding must still allow it.
    assert agent.grounding.check_args({"user_id": "abc-KEPT-123"}) == []
    # DROPPED id is no longer visible — grounding must refuse.
    assert agent.grounding.check_args({"target_id": "abc-DROPPED-999"}) != []


def test_replace_messages_preserves_harness_ids():
    """Harness-minted tool_call_ids address on-disk artifacts that
    outlive in-context truncation — they must survive a resync so
    the model can still call fetch_tool_output on them."""
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=ToolRegistry(),
        permissions=PermissionManager(auto_approve=True),
        llm=None,
    )
    agent.grounding.record_tool_call_id("call_abcdef1234567890")
    agent.replace_messages([])
    assert "call_abcdef1234567890" in agent.grounding.harness_ids
    assert agent.grounding.check_args(
        {"tool_call_id": "call_abcdef1234567890"}
    ) == []


def test_agent_reset_clears_tool_output_store(make_llm):
    from mouse.tools import tool_output_store

    tool_output_store.clear()
    registry = ToolRegistry()
    registry.register(Tool(
        name="echo", description="", parameters={},
        handler=lambda a: "payload",
        permission=PermissionLevel.SAFE,
    ))
    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[tool_call("echo", {}, tc_id="call_r")])),
        FakeResponse(FakeMessage(content="ok")),
    )
    agent = make_agent(llm, registry=registry)
    agent.chat("run echo")

    assert tool_output_store.size() >= 1
    agent.reset()
    assert tool_output_store.size() == 0


# ─── Pruner authoritative at dispatch ───────────────────────────────


def test_agent_refuses_dispatch_when_pruner_withheld_tool(make_llm):
    """If the model calls a tool the pruner didn't advertise on this
    step, the harness must refuse dispatch rather than silently
    consult the full registry. The real handler must never run."""
    from mouse.tools.pruner import PruneConfig, ToolPruner

    invoked: list[dict] = []

    registry = ToolRegistry()
    registry.register(Tool(
        name="bash", description="shell", parameters={},
        handler=lambda a: "", permission=PermissionLevel.SAFE,
    ))
    # A tool whose description shares no tokens with the user turn.
    registry.register(Tool(
        name="mcp_weather_forecast",
        description="Fetch a five-day weather forecast for a city.",
        parameters={},
        handler=lambda a: (invoked.append(a) or "72F"),
        permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        # Model invokes the hidden tool anyway.
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("mcp_weather_forecast", {"city": "SF"}, tc_id="c1"),
        ])),
        FakeResponse(FakeMessage(content="understood")),
    )
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        pruner=ToolPruner(PruneConfig(max_optional=5, min_score=1)),
    )

    # User message has no overlap with weather/forecast so the pruner
    # keeps only the core "bash" tool, and the weather tool is hidden.
    agent.chat("post a slack message about the release")

    # First LLM call must NOT have included the weather tool in its
    # schema list — otherwise this test wouldn't be exercising the
    # pruner-authoritative path.
    step1_names = {s["function"]["name"] for s in llm.calls[0]["tools"]}
    assert "mcp_weather_forecast" not in step1_names

    # Real handler was never called.
    assert invoked == []

    # A synthetic refusal result was appended for the model to see.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "HARNESS REFUSED" in tool_msgs[0]["content"]
    assert "not advertised" in tool_msgs[0]["content"]

    # The tool is now sticky, so a retry on the next step would succeed.
    assert "mcp_weather_forecast" in agent.sticky_tools


# ─── Session wiring ──────────────────────────────────────────────────


def test_agent_persists_user_assistant_and_tool_to_session(make_llm, tmp_path):
    """When a SessionManager is wired in, the Agent must hand off every
    user message, tool dispatch, and assistant reply so /resume and
    /search can reconstruct the turn."""
    from mouse.sessions import SessionManager

    calls: list[dict] = []

    registry = ToolRegistry()
    registry.register(Tool(
        name="echo", description="", parameters={},
        handler=lambda a: f"echoed:{a['x']}", permission=PermissionLevel.SAFE,
    ))

    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("echo", {"x": "abc"}, tc_id="t1"),
        ])),
        FakeResponse(FakeMessage(content="all done")),
    )

    session_mgr = SessionManager(base_dir=tmp_path / "sessions")
    record = session_mgr.create(project_root=str(tmp_path), model="fake-model")

    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        session=session_mgr,
    )
    agent.session_id = record.id

    # Capture calls by monkey-patching the real methods on this instance.
    real_ru = session_mgr.record_user
    real_ra = session_mgr.record_assistant
    real_rt = session_mgr.record_tool
    real_utc = session_mgr.update_token_count
    session_mgr.record_user = lambda c, **kw: calls.append(("user", c)) or real_ru(c, **kw)  # type: ignore[assignment]
    session_mgr.record_assistant = lambda c, **kw: calls.append(("assistant", c)) or real_ra(c, **kw)  # type: ignore[assignment]
    session_mgr.record_tool = lambda **kw: calls.append(("tool", kw["tool_name"], kw["tool_output"])) or real_rt(**kw)  # type: ignore[assignment]
    session_mgr.update_token_count = lambda t: calls.append(("tokens", t)) or real_utc(t)  # type: ignore[assignment]

    final = agent.chat("run the echo tool with x=abc")

    assert final == "all done"
    kinds = [c[0] for c in calls]
    assert kinds == ["user", "tool", "assistant", "tokens"]
    assert calls[0][1] == "run the echo tool with x=abc"
    assert calls[1][1] == "echo"
    assert calls[1][2] == "echoed:abc"
    assert calls[2][1] == "all done"

    session_mgr.close()


def test_agent_without_session_does_not_crash(make_llm):
    """The session kwarg is optional — the loop must still run clean
    when no SessionManager is attached (existing test paths rely on this)."""
    registry = ToolRegistry()
    llm = make_llm(FakeResponse(FakeMessage(content="hi")))
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=2),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
    )
    assert agent.session is None
    assert agent.chat("hello") == "hi"


# ─── Event log wiring ────────────────────────────────────────────────


def test_agent_emits_turn_events_to_file_backed_event_log(make_llm, tmp_path):
    """With a file-backed EventLog attached, a single chat turn must
    write at least the turn.start and turn.end decisions. Downstream
    consumers (session replay, analytics) rely on the loop bookends
    being present even for a pure-text turn with no tool calls."""
    from mouse.engine.events import EventLog

    events_path = tmp_path / "events.jsonl"
    event_log = EventLog(path=events_path)

    llm = make_llm(FakeResponse(FakeMessage(content="done")))
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=2),
        tools=ToolRegistry(),
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        events=event_log,
    )

    agent.chat("anything")
    event_log.close()

    assert events_path.exists(), "event log file was never written"
    lines = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if line.strip()
    ]
    types = [rec["type"] for rec in lines]
    assert "turn.start" in types
    assert "turn.end" in types
    # llm.response must fire for the one step that ran.
    assert "llm.response" in types


def test_agent_records_tool_dispatch_outcome_in_events(make_llm, tmp_path):
    """A successful tool call lands in the event log with outcome=ok
    and a non-negative duration_ms so latency histograms can be built
    directly from the JSONL without re-deriving it from transcripts."""
    from mouse.engine.events import EventLog

    registry = ToolRegistry()
    registry.register(Tool(
        name="echo", description="", parameters={},
        handler=lambda a: f"echoed:{a['x']}", permission=PermissionLevel.SAFE,
    ))
    llm = make_llm(
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("echo", {"x": "abc"}, tc_id="t1"),
        ])),
        FakeResponse(FakeMessage(content="done")),
    )

    events_path = tmp_path / "events.jsonl"
    event_log = EventLog(path=events_path)

    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=3),
        tools=registry,
        permissions=PermissionManager(auto_approve=True),
        llm=llm,
        events=event_log,
    )
    agent.chat("run echo with x=abc")
    event_log.close()

    dispatch_events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line)["type"] == "tool.dispatch"
    ]
    assert len(dispatch_events) == 1
    ev = dispatch_events[0]
    assert ev["tool"] == "echo"
    assert ev["tc_id"] == "t1"
    assert ev["outcome"] == "ok"
    assert ev["duration_ms"] >= 0
    assert ev["output_chars"] == len("echoed:abc")


def test_null_event_log_is_silent_and_safe(make_llm):
    """An agent built without an EventLog kwarg falls back to the null
    sink — chatting never touches the filesystem and every decision
    the loop would emit is silently dropped."""
    agent = Agent(
        config=AgentConfig(model="fake-model", max_steps=2),
        tools=ToolRegistry(),
        permissions=PermissionManager(auto_approve=True),
        llm=make_llm(FakeResponse(FakeMessage(content="ok"))),
    )
    # The default events sink is non-None but path-less.
    assert agent.events is not None
    assert agent.events.path is None
    agent.chat("ping")  # must not raise
