"""Minimal HTTP API for Electron clients."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from mouse.config import load_config
from mouse.project import load_project_context, PROJECT_ROOT
from mouse.tools.builtin import build_default_tools
from mouse.tools.permissions import PermissionManager
from mouse.tools.pruner import ToolPruner
from mouse.mcp.manager import load_mcp_servers
from mouse.engine.agent import Agent, AgentConfig
from mouse.engine.events import EventLog
from mouse.skills import discover_skills
from mouse.sessions import SessionManager
from mouse.memory.manager import MemoryManager


def build_runtime() -> tuple[Agent, SessionManager]:
    config = load_config()
    tools = build_default_tools()
    permissions = PermissionManager(auto_approve=config.auto_approve)
    skills = discover_skills()
    pruner = ToolPruner()
    memory = MemoryManager()
    session_mgr = SessionManager()
    # API mode defers session creation to explicit /session/create;
    # until then the agent runs with a null event sink. attach_event_log
    # below swaps in a real file sink whenever a session is created or
    # resumed.
    event_log: EventLog = EventLog.null()

    try:
        mcp_mgr = load_mcp_servers(tools)
    except Exception as exc:
        print(f"Warning: MCP load failed: {exc}")
        mcp_mgr = None

    agent = Agent(
        AgentConfig(
            model=config.model,
            max_steps=config.max_steps,
            max_retries=config.max_retries,
            auto_approve=config.auto_approve,
            system_prompt=config.system_prompt,
            project_context=load_project_context(),
        ),
        tools,
        permissions,
        skills=skills,
        pruner=pruner,
        memory=memory,
        session=session_mgr,
        events=event_log,
    )
    agent._mcp_mgr = mcp_mgr
    return agent, session_mgr


def attach_event_log(agent: Agent, session_mgr: SessionManager, session_id: str) -> None:
    """Close the agent's current event sink and reattach a file-backed
    one pointing at the new session's events.jsonl. Called on every
    session switch so each session's audit trail lands in its own file.
    """
    old = getattr(agent, "events", None)
    if old is not None:
        old.close()
    agent.events = EventLog(path=session_mgr.events_path(session_id))


def summarize_agent(agent: Agent, session_mgr: SessionManager):
    active = session_mgr.active
    return {
        "model": agent.llm.model,
        "auto_approve": agent.permissions.auto_approve,
        "project_root": PROJECT_ROOT,
        "stats": agent.stats.summary(),
        "messages": len(agent.messages),
        "session": None if active is None else {
            "id": active.id,
            "title": active.title,
            "status": active.status,
            "created_at": active.created_at,
            "updated_at": active.updated_at,
        },
        "tools": [
            {"name": n, "permission": agent.tools.get(n).permission.value, "description": agent.tools.get(n).description}
            for n in agent.tools.list_names()
        ],
        "skills": [
            {"name": s.name, "tier": s.tier, "description": s.description}
            for s in agent.skills.all()
        ],
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "updated_at": s.updated_at,
                "status": s.status,
                "model": s.model,
                "total_tokens": s.total_tokens,
                "summary": s.summary,
            }
            for s in session_mgr.list_sessions(limit=20)
        ],
    }


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    agent, session_mgr = build_runtime()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/state":
                self._send(200, summarize_agent(agent, session_mgr))
            elif path == "/sessions":
                self._send(200, {"sessions": summarize_agent(agent, session_mgr)["sessions"]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/chat":
                message = (body.get("message") or "").strip()
                if not message:
                    self._send(400, {"error": "message required"})
                    return
                try:
                    reply = agent.chat(message)
                    self._send(200, {"reply": reply, "state": summarize_agent(agent, session_mgr)})
                except Exception as e:
                    self._send(500, {"error": str(e)})
            elif path == "/session/create":
                rec = session_mgr.create(project_root=PROJECT_ROOT, model=agent.llm.model, title=body.get("title", ""))
                agent.session_id = rec.id
                agent.session_title = rec.title
                attach_event_log(agent, session_mgr, rec.id)
                self._send(200, {"session": rec.__dict__, "state": summarize_agent(agent, session_mgr)})
            elif path == "/session/resume":
                rec = session_mgr.resume(body.get("id", ""))
                agent.session_id = rec.id
                agent.session_title = rec.title
                attach_event_log(agent, session_mgr, rec.id)
                self._send(200, {"session": rec.__dict__, "state": summarize_agent(agent, session_mgr)})
            else:
                self._send(404, {"error": "not found"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"status": "ready", "host": host, "port": port}))
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
