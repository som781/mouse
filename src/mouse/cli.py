"""CLI entry point: main loop, slash commands, welcome banner."""

from __future__ import annotations

import os
import sys

from mouse.term import C, term_width
from mouse.project import PROJECT_ROOT, load_project_context
from mouse.config import load_config
from mouse.errors import MouseError
from mouse.logging import get_logger
from mouse.tools.builtin import build_default_tools
from mouse.tools.permissions import PermissionManager
from mouse.tools.pruner import ToolPruner
from mouse.mcp.manager import load_mcp_servers
from mouse.engine.agent import Agent, AgentConfig
from mouse.engine.events import EventLog
from mouse.skills import discover_skills
from mouse.sessions import SessionManager
from mouse.sessions.commands import handle_slash_session
from mouse.sessions.compaction import compact as adaptive_compact
from mouse.sessions.handoff import handle_progress
from mouse.memory.manager import MemoryManager
from mouse.memory.flush import compact_with_flush

log = get_logger("mouse.cli")


if os.environ.get("MOUSE_ELECTRON_BRIDGE") == "1":
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse

    class _BridgeState:
        agent = None
        session_mgr = None

    def _agent_summary(agent):
        return {
            "model": agent.llm.model,
            "stats": agent.stats.summary(),
            "messages": len(agent.messages),
            "auto_approve": agent.permissions.auto_approve,
            "tools": [
                {
                    "name": n,
                    "permission": agent.tools.get(n).permission.value,
                    "description": agent.tools.get(n).description,
                }
                for n in agent.tools.list_names()
            ],
            "skills": [
                {"name": s.name, "tier": s.tier, "description": s.description}
                for s in agent.skills.all()
            ],
        }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if urlparse(self.path).path == "/state":
                self._send(200, _agent_summary(_BridgeState.agent))
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/chat":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            message = body.get("message", "")
            if not message.strip():
                self._send(400, {"error": "message required"})
                return
            try:
                output = _BridgeState.agent.chat(message)
                self._send(200, {"reply": output, "state": _agent_summary(_BridgeState.agent)})
            except Exception as e:
                self._send(500, {"error": str(e)})

    # bridge mode is handled after initialization in main()


HELP_TEXT = """\
  Available commands:
    /help             Show this help message
    /reset            Clear conversation history (start fresh)
    /stats            Show token usage and stats
    /model            Show or change the current model
    /compact          Summarize conversation to save context
    /tools            List available tools
    /skills           List discovered skills (builtin/project/user)
    /mcp              Show connected MCP servers and their tools
    /auto             Toggle auto-approve mode
    /new [title]      Start a new session (clean context)
    /resume <id>      Resume a previous session
    /sessions         List recent sessions
    /search <query>   Full-text search across all sessions
    /progress         Generate a progress.md handoff file
    quit              Exit the agent
"""


def handle_slash_command(command: str, agent: Agent, session_mgr: SessionManager | None = None) -> bool:
    """Handle slash commands. Returns True if handled."""
    cmd = command.strip().lower()

    # Session commands are handled by their own dispatcher so the CLI
    # glue stays thin. It returns the pre-formatted message to print,
    # or None if the command isn't session-related.
    if session_mgr is not None:
        session_out = handle_slash_session(command, session_mgr, agent)
        if session_out is not None:
            color = C.RED if session_out.startswith("✗") else C.CYAN
            print(C.styled(f"  {session_out}", color))
            return True

    if cmd == "/help":
        print(C.styled(HELP_TEXT, C.DIM))
        return True

    elif cmd == "/reset":
        agent.reset()
        print(C.styled("  ✓ Conversation cleared", C.GREEN))
        return True

    elif cmd == "/stats":
        print(C.styled(f"  {agent.stats.summary()}", C.CYAN))
        print(C.styled(f"  Messages in context: {len(agent.messages)}", C.DIM))
        return True

    elif cmd.startswith("/model"):
        parts = command.split(maxsplit=1)
        if len(parts) > 1:
            agent.llm.set_model(parts[1].strip())
            print(C.styled(f"  ✓ Model changed to: {agent.llm.model}", C.GREEN))
        else:
            print(C.styled(f"  Current model: {agent.llm.model}", C.CYAN))
        return True

    elif cmd == "/compact":
        msg_count = len(agent.messages)
        if msg_count < 4:
            print(C.styled("  Not enough history to compact", C.YELLOW))
            return True

        print(C.styled("  Compacting conversation (adaptive)...", C.DIM))
        try:
            # When a memory manager is attached, compact+flush distils
            # the discarded history into episodic memory before the
            # summarizer rewrites it. Otherwise fall back to a plain
            # compaction — the in-memory pruning is the same.
            if agent.memory is not None:
                new_msgs, report, recorded = compact_with_flush(
                    agent.messages,
                    agent.llm,
                    agent.memory,
                    session_id=agent.session_id,
                    session_title=agent.session_title,
                )
            else:
                new_msgs, report = adaptive_compact(agent.messages, agent.llm)
                recorded = []
        except Exception as e:
            print(C.styled(f"  ✗ Compaction failed: {e}", C.RED))
            return True

        # Route through replace_messages so grounding is resynced to
        # the trimmed context — otherwise grounding could approve ids
        # the model can no longer see.
        agent.replace_messages(new_msgs)
        print(C.styled(
            f"  ✓ {report.original_tokens} → {report.final_tokens} tokens "
            f"(pruned {report.tool_messages_pruned} tool outputs, "
            f"summarized {report.messages_summarized} messages)",
            C.GREEN,
        ))
        if recorded:
            print(C.styled(
                f"  📝 Flushed {len(recorded)} observations to episodic memory",
                C.DIM,
            ))
        if report.summary_text:
            print(C.styled(f"  Summary: {report.summary_text[:200]}...", C.DIM))
        return True

    elif cmd == "/tools":
        print(C.styled("  Available tools:", C.CYAN))
        for name in agent.tools.list_names():
            tool = agent.tools.get(name)
            perm = tool.permission.value
            prefix = "🔌" if name.startswith("mcp_") else "🔧"
            print(C.styled(f"    {prefix} {name}", C.BOLD) + C.styled(f"  [{perm}]  {tool.description[:70]}", C.DIM))
        return True

    elif cmd == "/skills":
        skills = agent.skills.all()
        if not skills:
            print(C.styled("  No skills discovered.", C.YELLOW))
            return True
        print(C.styled(f"  Skills ({len(skills)}):", C.CYAN))
        # Group by tier so the precedence is visible.
        by_tier: dict[str, list] = {}
        for s in skills:
            by_tier.setdefault(s.tier, []).append(s)
        for tier in ("builtin", "project", "user"):
            group = by_tier.get(tier, [])
            if not group:
                continue
            print(C.styled(f"    [{tier}]", C.BOLD))
            for s in group:
                marker = " ✨" if s in agent.active_skills else ""
                print(C.styled(f"      • {s.name}{marker}", C.BOLD)
                      + C.styled(f"  {s.description[:70]}", C.DIM))
        return True

    elif cmd == "/mcp":
        mcp_tools = [n for n in agent.tools.list_names() if n.startswith("mcp_")]
        if not mcp_tools:
            print(C.styled("  No MCP servers connected.", C.YELLOW))
            print(C.styled("  Add a mcp.json or mcp_servers.json to your project root, or set MCP_CONFIG env var.", C.DIM))
        else:
            print(C.styled(f"  MCP tools ({len(mcp_tools)}):", C.CYAN))
            for name in mcp_tools:
                tool = agent.tools.get(name)
                print(C.styled(f"    🔌 {name}", C.BOLD))
                print(C.styled(f"       {tool.description}", C.DIM))
        return True

    elif cmd == "/progress":
        if session_mgr is None:
            print(C.styled("  ✗ Session manager not attached", C.RED))
            return True
        out = handle_progress(session_mgr, agent)
        color = C.RED if out.startswith("✗") else C.GREEN
        print(C.styled(f"  {out}", color))
        return True

    elif cmd == "/auto":
        agent.permissions.auto_approve = not agent.permissions.auto_approve
        state = "ON" if agent.permissions.auto_approve else "OFF"
        color = C.YELLOW if agent.permissions.auto_approve else C.GREEN
        print(C.styled(f"  Auto-approve is now {state}", color))
        return True

    return False


def main():
    if os.environ.get("MOUSE_API_SERVER") == "1":
        from mouse.api.server import run_server
        run_server()
        return

    # Load user/project config (defaults applied for missing values)
    try:
        user_config = load_config()
    except MouseError as e:
        log.error("config: %s", e)
        sys.exit(1)

    # Honor MCP_CONFIG override from config file (env var still wins via load_mcp_servers)
    if user_config.mcp_config_path and not os.environ.get("MCP_CONFIG"):
        os.environ["MCP_CONFIG"] = user_config.mcp_config_path

    # Load project context
    project_context = load_project_context()

    # Build components
    config = AgentConfig(
        model=user_config.model,
        max_steps=user_config.max_steps,
        max_retries=user_config.max_retries,
        auto_approve=user_config.auto_approve,
        system_prompt=user_config.system_prompt,
        project_context=project_context,
    )
    tools = build_default_tools()

    # Load MCP servers (if configured)
    mcp_manager = load_mcp_servers(tools)

    permissions = PermissionManager(auto_approve=user_config.auto_approve)

    # Discover skills from all three tiers (builtin/project/user).
    skills = discover_skills()

    # Dynamic tool pruner trims irrelevant MCP tools per turn.
    pruner = ToolPruner()

    # Persistent memory layers (episodic / preferences / semantic).
    # Failure here is non-fatal: the agent still runs, just without
    # cross-session recall.
    try:
        memory = MemoryManager()
    except Exception as e:
        log.warning("memory: disabled (%s)", e)
        memory = None

    # Open the session manager first so the Agent can persist every
    # user turn, tool call, and assistant reply as it happens. Without
    # this wiring, /progress, /resume and /search see an empty log.
    session_mgr = SessionManager()
    record = session_mgr.create(
        project_root=str(PROJECT_ROOT),
        model=user_config.model,
    )

    # File-backed event log for this session. Every harness decision
    # (pruner picks, grounding refusals, permission denies, tool
    # latencies) lands in the session dir alongside the tool-output
    # store so a single session id locates the full audit trail.
    event_log = EventLog(path=session_mgr.events_path(record.id))

    agent = Agent(
        config, tools, permissions,
        skills=skills, pruner=pruner, memory=memory, session=session_mgr,
        events=event_log,
    )
    agent.session_id = record.id
    agent.session_title = record.title or ""

    # Welcome
    width = term_width()
    print(C.styled(f"\n{'═' * width}", C.CYAN))
    print(C.styled("  🤖 Mouse — AI Agent Harness", C.BOLD, C.CYAN))
    print(C.styled(f"  Model: {user_config.model}", C.DIM))
    print(C.styled(f"  Project: {PROJECT_ROOT}", C.DIM))
    print(C.styled(f"  Tools: {', '.join(tools.list_names())}", C.DIM))
    if len(skills) > 0:
        print(C.styled(f"  Skills: {', '.join(skills.names())}", C.DIM))
    if mcp_manager and mcp_manager.connections:
        servers = ", ".join(mcp_manager.connections.keys())
        print(C.styled(f"  MCP Servers: {servers}", C.DIM))
    print(C.styled("  Type /help for commands", C.DIM))
    print(C.styled(f"{'═' * width}\n", C.CYAN))

    while True:
        try:
            user_input = input(C.styled("🧑 You: ", C.BOLD)).strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print(C.styled(f"\n  📊 Session: {agent.stats.summary()}", C.DIM))
                session_mgr.close()
                event_log.close()
                if mcp_manager:
                    mcp_manager.shutdown()
                print(C.styled("  Goodbye! 👋\n", C.CYAN))
                break

            # Handle slash commands
            if user_input.startswith("/"):
                prev_session = agent.session_id
                if handle_slash_command(user_input, agent, session_mgr=session_mgr):
                    # /new and /resume rewrite agent.session_id; when
                    # that happens, close the old JSONL sink and open
                    # one under the new session dir so every audit
                    # trail lives next to the tool outputs it describes.
                    if agent.session_id and agent.session_id != prev_session:
                        event_log.close()
                        event_log = EventLog(
                            path=session_mgr.events_path(agent.session_id)
                        )
                        agent.events = event_log
                    continue

            # Run the agent
            agent.chat(user_input)

            # After the first exchange on a fresh session, ask the LLM
            # for a short title so /sessions shows something meaningful.
            if not agent.session_title:
                try:
                    title = session_mgr.auto_title(agent.llm)
                    if title:
                        agent.session_title = title
                except Exception as e:
                    log.debug("auto_title failed: %s", e)

        except KeyboardInterrupt:
            print(C.styled("\n  (Ctrl+C) Type 'quit' to exit", C.DIM))
        except EOFError:
            break


if __name__ == "__main__":
    main()
