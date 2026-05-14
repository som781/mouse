"""The 6 built-in tools: bash, read_file, write_file, search_files, list_directory, python."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from mouse.term import C
from mouse.project import PROJECT_ROOT
from mouse.tools.registry import PermissionLevel, Tool, ToolRegistry
from mouse.tools import tool_output_store


def handle_bash(args: dict) -> str:
    """Execute a shell command with live output for long-running processes.

    The timeout is whatever the caller passed (clamped to the 300 s
    hard ceiling). The harness deliberately does not sniff command
    strings for specific tool names — a caller that expects a long
    install should pass ``timeout`` explicitly. Keeping this generic
    is what lets mouse stay a harness rather than a build assistant.
    """
    command = args["command"]
    timeout = min(args.get("timeout", 120), 300)

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "TERM": "dumb"},
        )

        # Stream output line-by-line so user sees progress
        output_lines = []

        stderr_lines = []

        def read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        start = time.time()
        last_status = start

        for line in process.stdout:
            output_lines.append(line)
            elapsed = time.time() - start

            # Show progress dots every 5 seconds for long commands
            if elapsed - (last_status - start) > 5 and elapsed > 5:
                print(C.styled(f"  │   ... ({elapsed:.0f}s elapsed, {len(output_lines)} lines)", C.DIM))
                last_status = start + elapsed

            if elapsed > timeout:
                process.kill()
                output_lines.append(f"\n[KILLED: exceeded {timeout}s timeout]")
                break

        process.wait(timeout=10)
        stderr_thread.join(timeout=5)

        output = "".join(output_lines)
        if stderr_lines:
            output += "\n[STDERR]\n" + "".join(stderr_lines)
        if process.returncode and process.returncode != 0:
            output += f"\n[exit code: {process.returncode}]"

        return output.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        process.kill()
        return f"ERROR: Command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def handle_read_file(args: dict) -> str:
    """Read a file's contents."""
    path = args["path"]
    try:
        content = Path(path).read_text()
        lines = content.split("\n")
        if len(lines) > 500:
            return (
                f"[File has {len(lines)} lines. Showing first 200 + last 50]\n\n"
                + "\n".join(lines[:200])
                + f"\n\n... [{len(lines) - 250} lines omitted] ...\n\n"
                + "\n".join(lines[-50:])
            )
        return content
    except Exception as e:
        return f"ERROR: {e}"


def handle_write_file(args: dict) -> str:
    """Write content to a file."""
    path = args["path"]
    content = args["content"]
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lines = content.count("\n") + 1
        return f"✓ Wrote {len(content)} chars ({lines} lines) to {path}"
    except Exception as e:
        return f"ERROR: {e}"


_SEARCH_EXCLUDES = (
    ".git", "node_modules", "venv", "__pycache__",
    ".venv", "dist", "build",
)


def handle_search_files(args: dict) -> str:
    """Search for a pattern across files using grep.

    Arguments are passed as an argv list (shell=False) so pattern/path
    values that happen to contain shell metacharacters can't execute
    arbitrary commands.
    """
    pattern = args["pattern"]
    path = args.get("path", PROJECT_ROOT)
    file_type = args.get("file_type", "")

    argv = ["grep", "-rn"]
    for d in _SEARCH_EXCLUDES:
        argv.append(f"--exclude-dir={d}")
    if file_type:
        argv.append(f"--include=*{file_type}")
    argv.extend(["--", pattern, path])

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
        )
        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{pattern}' in {path}"
        lines = output.split("\n")
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n\n... and {len(lines) - 50} more matches"
        return output
    except Exception as e:
        return f"ERROR: {e}"


_LIST_DIR_EXCLUDES = ("node_modules", ".git", "venv", "__pycache__")


def handle_list_directory(args: dict) -> str:
    """List directory contents (up to 2 levels deep).

    Runs ``find`` with shell=False so a path containing metacharacters
    can't execute arbitrary commands.
    """
    path = str(args.get("path", PROJECT_ROOT))
    argv = ["find", path, "-maxdepth", "2"]
    for d in _LIST_DIR_EXCLUDES:
        argv += ["-not", "-path", f"*/{d}/*"]
    argv += ["-not", "-name", ".*"]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=10, cwd=PROJECT_ROOT
        )
        lines = sorted((result.stdout or "").splitlines())[:100]
        return "\n".join(lines).strip() or "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"


def handle_fetch_tool_output(args: dict) -> str:
    """Return a slice of a previously-truncated tool output."""
    tc_id = args["tool_call_id"]
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", 4000))
    return tool_output_store.fetch(tc_id, offset=offset, limit=limit)


def handle_search_tool_output(args: dict) -> str:
    """Grep a previously-truncated tool output for a regex."""
    tc_id = args["tool_call_id"]
    pattern = args["pattern"]
    max_matches = int(args.get("max_matches", 20))
    context = int(args.get("context", 1))
    return tool_output_store.search(
        tc_id, pattern, max_matches=max_matches, context=context
    )


def _format_preview_shape(preview: dict) -> str:
    """Render a manifest preview into a one-line shape descriptor."""
    fmt = preview.get("format", "?")
    if fmt == "json_list":
        shape = f"json_list[{preview.get('len', '?')}]"
        keys = preview.get("sample_keys") or []
        if keys:
            shape += f" sample_keys={','.join(keys[:5])}"
        return shape
    if fmt == "json_object":
        keys = preview.get("keys") or []
        return f"json_object keys={','.join(keys[:5])}"
    if fmt == "ndjson":
        return f"ndjson[{preview.get('lines', '?')} lines]"
    if fmt == "text":
        return f"text[{preview.get('lines', '?')} lines]"
    if fmt in ("error", "denied", "empty"):
        return fmt
    return fmt


def handle_list_tool_outputs(args: dict) -> str:
    """List the saved tool outputs in this session, newest first.

    The model uses this to discover what artifacts already exist on
    disk — full payloads from earlier tool calls that were truncated
    in-context. Each row carries a one-line shape descriptor (json
    list of N items, ndjson, text, …), the total size, the
    tool_call_id (for fetch/search), and the on-disk path (so any
    path-taking tool can process it directly).
    """
    tool_name = (args.get("tool_name") or "").strip() or None
    limit = int(args.get("limit", 20))
    rows = tool_output_store.list_outputs(tool_name=tool_name, limit=limit)
    if not rows:
        if tool_name:
            return f"No saved tool outputs match tool_name={tool_name!r}."
        return "No saved tool outputs in this session yet."

    out: list[str] = [f"[{len(rows)} saved tool outputs]"]
    for r in rows:
        shape = _format_preview_shape(r.get("preview") or {})
        args_str = ", ".join(
            f"{k}={v!r}" for k, v in (r.get("args") or {}).items()
        )
        if len(args_str) > 80:
            args_str = args_str[:77] + "..."
        path = r.get("path") or "(memory backend — no file)"
        out.append(
            f"  • {r.get('tool_name') or '?'}({args_str}) → "
            f"{r.get('total_chars', 0):,} chars, {shape}\n"
            f"    tool_call_id={r.get('tc_id', '?')}\n"
            f"    path={path}"
        )
    return "\n".join(out)


def handle_python(args: dict) -> str:
    """Execute a Python snippet and return the output."""
    code = args["code"]
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Script timed out"
    except Exception as e:
        return f"ERROR: {e}"


def build_default_tools() -> ToolRegistry:
    """Create registry with all built-in tools."""

    registry = ToolRegistry()

    registry.register(Tool(
        name="bash",
        description=(
            "Run a shell command from the project root. "
            "Default timeout is 120 s; pass a larger 'timeout' (up to "
            "300 s) for commands that may run longer, such as installs "
            "or builds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 300)"},
            },
            "required": ["command"],
        },
        handler=handle_bash,
        permission=PermissionLevel.ASK,
    ))

    registry.register(Tool(
        name="read_file",
        description="Read the contents of a file. Large files are automatically truncated.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"},
            },
            "required": ["path"],
        },
        handler=handle_read_file,
        permission=PermissionLevel.SAFE,
    ))

    registry.register(Tool(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
        handler=handle_write_file,
        permission=PermissionLevel.ASK,
    ))

    registry.register(Tool(
        name="search_files",
        description=(
            "Search for a text pattern across files (grep). "
            "Searches the entire project by default. Use absolute paths to search elsewhere. "
            "Excludes .git, node_modules, venv, __pycache__ automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search in (default: project root). Use absolute paths for broader searches."},
                "file_type": {"type": "string", "description": "File extension filter, e.g. '.py', '.ts'"},
            },
            "required": ["pattern"],
        },
        handler=handle_search_files,
        permission=PermissionLevel.SAFE,
    ))

    registry.register(Tool(
        name="list_directory",
        description="List files and directories (up to 2 levels deep, excludes node_modules and .git).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current dir)"},
            },
        },
        handler=handle_list_directory,
        permission=PermissionLevel.SAFE,
    ))

    registry.register(Tool(
        name="fetch_tool_output",
        description=(
            "Fetch a slice of the FULL output of a previous tool call. "
            "Use this when a prior tool's result shows a '[...truncated N chars...]' "
            "marker and you need to see the trimmed content. "
            "Pass the tool_call_id from the earlier tool message, plus optional "
            "offset/limit to paginate through large outputs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "ID of the earlier tool call whose full output you want.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Start character offset (default 0).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max chars to return (default 4000).",
                },
            },
            "required": ["tool_call_id"],
        },
        handler=handle_fetch_tool_output,
        permission=PermissionLevel.SAFE,
    ))

    registry.register(Tool(
        name="search_tool_output",
        description=(
            "Regex-search the FULL output of a previous tool call and return "
            "matching lines with surrounding context. Use this when a tool's "
            "result was truncated and you need to find a specific value inside "
            "the omitted portion without paging through the whole thing."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "ID of the earlier tool call to search.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Python regex to match against each line.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Max matches to return (default 20).",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context above/below each match (default 1).",
                },
            },
            "required": ["tool_call_id", "pattern"],
        },
        handler=handle_search_tool_output,
        permission=PermissionLevel.SAFE,
    ))

    registry.register(Tool(
        name="list_tool_outputs",
        description=(
            "List the FULL outputs of earlier tool calls that have been "
            "saved in this session. Each row carries a one-line shape "
            "descriptor (e.g. 'json_list[47] sample_keys=id,title,status'), "
            "the total size, the tool_call_id (use with fetch_tool_output / "
            "search_tool_output), and the on-disk path (use with read_file "
            "or any path-taking tool). Use this to discover what artifacts "
            "you already have before re-running an expensive tool call, or "
            "to find an earlier result by intent rather than scrolling back "
            "through the conversation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Optional: filter to outputs from a specific tool name.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 20, newest first).",
                },
            },
        },
        handler=handle_list_tool_outputs,
        permission=PermissionLevel.SAFE,
    ))

    registry.register(Tool(
        name="python",
        description="Execute a Python code snippet and return its stdout/stderr output.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
        handler=handle_python,
        permission=PermissionLevel.ASK,
    ))

    return registry
