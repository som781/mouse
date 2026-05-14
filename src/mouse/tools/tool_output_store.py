"""Persistent store for full (pre-truncation) tool outputs.

When the agent truncates a tool result before appending it to the
conversation (see ``MAX_TOOL_RESULT_CHARS`` in ``engine/agent.py``), the
*full* text is kept here, keyed by tool_call_id. The ``fetch_tool_output``
and ``search_tool_output`` built-in tools let the model drill back into
the trimmed portion without bloating every subsequent turn.

**Two backends, same API:**

1. **In-memory (default)** — a module-level dict. Used by unit tests and
   by the agent when no session is active. Fast, ephemeral.

2. **Disk** — files under ``<session_dir>/tool_outputs/<tc_id>.txt``.
   Activated by calling :func:`configure` with a session directory. This
   is the Sprint 3 context-firewall backend: outputs survive process
   restart, show up in session replay, and can be read directly by
   episodic memory (Sprint 4) without going through this module.

Callers do not care which backend is active. ``save`` / ``fetch`` /
``search`` / ``clear`` / ``size`` all work identically either way. The
:class:`Agent` switches modes by calling :func:`configure` when it
starts/ends a session; everything else keeps calling the same four
functions.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# ─── Backend state ──────────────────────────────────────────────────

# In-memory fallback: tc_id -> full output.
_memory: dict[str, str] = {}

# Parallel manifest dict, populated on every save regardless of backend.
# When the disk backend is active, the same manifest is also written as
# a ``<tc_id>.json`` sidecar so it survives process restart and is
# discoverable by future sessions.
_memory_meta: dict[str, dict] = {}

# When set, save/fetch/search operate against this directory instead of
# ``_memory``. The directory is created on first save.
_session_dir: Path | None = None

# Don't try to JSON-parse payloads bigger than this for the preview —
# the format sniff is supposed to be cheap.
_PREVIEW_MAX_PARSE_CHARS = 1_000_000

# Cap the per-item directory we build for json_list previews. The
# directory exists so the model can resolve a name → id lookup even
# when the full payload was truncated; >50 items per list rarely
# needs to be visible all at once.
_INDEX_MAX_ITEMS = 50

# Heuristic key names used to pick a "label" and "id" for each item
# in a json_list, in priority order. Generic enough to catch most
# structured collection responses (orgs, users, projects, tickets…).
_LABEL_KEY_HINTS = (
    "name", "org_name", "group_name", "full_name", "display_name",
    "title", "label", "subject", "summary",
)
_ID_KEY_HINTS = ("id", "uuid", "slug", "key")


def configure(session_dir: str | Path | None) -> None:
    """Switch the backend to disk (when given a path) or memory (None).

    Switching backends does NOT migrate already-stored outputs — the
    agent calls this at session start/end, so each session has a fresh
    slate. Callers that need to carry data across a switch should
    ``fetch`` it first.
    """
    global _session_dir
    if session_dir is None:
        _session_dir = None
        return
    _session_dir = Path(session_dir).expanduser()
    # Defer mkdir to first save so probing configure() is side-effect-free.


def _tool_outputs_dir() -> Path | None:
    if _session_dir is None:
        return None
    return _session_dir / "tool_outputs"


def _safe_id(tc_id: str) -> str:
    """Sanitize a tool_call_id for use as a filename.

    OpenAI IDs are alphanumeric with underscores, but a defensive strip
    protects against an MCP server that invents something exotic.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tc_id)[:200]


def _path_for(tc_id: str) -> Path | None:
    d = _tool_outputs_dir()
    if d is None:
        return None
    return d / f"{_safe_id(tc_id)}.txt"


def _meta_path_for(tc_id: str) -> Path | None:
    d = _tool_outputs_dir()
    if d is None:
        return None
    return d / f"{_safe_id(tc_id)}.json"


def path_for(tc_id: str) -> str | None:
    """Return the on-disk path of the saved output, or None when the
    memory backend is active. Used by the agent to embed the path in
    the truncation marker so the model can read or process the file
    directly.
    """
    p = _path_for(tc_id)
    return str(p) if p is not None else None


# ─── Format-aware preview ───────────────────────────────────────────


def _pick_id_field(item: dict) -> tuple[str, str] | tuple[None, None]:
    """Return ``(key, value)`` for an id-shaped field in ``item``."""
    # Exact-name hits first (id, uuid, slug, key).
    for hint in _ID_KEY_HINTS:
        if hint in item:
            v = item[hint]
            if isinstance(v, (str, int)) and str(v):
                return hint, str(v)
    # Then anything that ends in ``_id`` (org_id, group_id, …).
    for k, v in item.items():
        if isinstance(k, str) and k.lower().endswith("_id"):
            if isinstance(v, (str, int)) and str(v):
                return k, str(v)
    return None, None


def _pick_label_field(item: dict) -> tuple[str, str] | tuple[None, None]:
    """Return ``(key, value)`` for a human-readable label in ``item``."""
    for hint in _LABEL_KEY_HINTS:
        if hint in item:
            v = item[hint]
            if isinstance(v, str) and v.strip():
                return hint, v.strip()[:120]
    # Fallback: any short non-id string field.
    for k, v in item.items():
        if not isinstance(k, str) or k.lower().endswith("_id"):
            continue
        if isinstance(v, str) and 0 < len(v) < 120:
            return k, v.strip()
    return None, None


def _build_items_index(parsed: list) -> list[dict]:
    """Build a compact ``[(id, label)]`` directory of a json_list payload.

    The index is what makes a truncated list still discoverable: even
    if the model only sees the first item in the truncated content,
    the marker can carry the directory of every item so a name → id
    lookup ("which one is Globex?") resolves without needing to
    re-fetch the whole payload.
    """
    out: list[dict] = []
    for raw in parsed[:_INDEX_MAX_ITEMS]:
        if not isinstance(raw, dict):
            # Non-dict items: just stringify briefly.
            out.append({"value": str(raw)[:80]})
            continue
        id_key, id_val = _pick_id_field(raw)
        label_key, label_val = _pick_label_field(raw)
        entry: dict = {}
        if id_val is not None:
            entry["id_key"] = id_key
            entry["id"] = id_val
        if label_val is not None:
            entry["label_key"] = label_key
            entry["label"] = label_val
        if entry:
            out.append(entry)
    return out


def _format_aware_preview(content: str) -> dict:
    """Sniff ``content`` and return a tiny structural summary.

    The goal is *not* to summarize semantically — it's to tell the
    model "this artifact is a JSON list of 47 items with these keys"
    so it can decide whether the saved file is worth re-reading or
    piping into another tool. No LLM calls, no expensive parsing.
    """
    if not content:
        return {"format": "empty"}

    stripped = content.lstrip()
    head_lower = stripped[:20].lower()
    if head_lower.startswith("error:"):
        return {
            "format": "error",
            "first_line": content.splitlines()[0][:200] if content.splitlines() else "",
        }
    if head_lower.startswith("denied:"):
        return {
            "format": "denied",
            "first_line": content.splitlines()[0][:200] if content.splitlines() else "",
        }

    # Whole-document JSON?
    first_char = stripped[:1]
    if first_char in "[{" and len(content) <= _PREVIEW_MAX_PARSE_CHARS:
        try:
            parsed = json.loads(stripped)
        except (ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            preview: dict[str, Any] = {"format": "json_list", "len": len(parsed)}
            if parsed and isinstance(parsed[0], dict):
                preview["sample_keys"] = list(parsed[0].keys())[:8]
            # Build a compact directory of (id, label) tuples so a
            # truncated list is still searchable by name.
            index = _build_items_index(parsed)
            if index:
                preview["items_index"] = index
                preview["items_index_truncated"] = len(parsed) > len(index)
            return preview
        if isinstance(parsed, dict):
            return {
                "format": "json_object",
                "keys": list(parsed.keys())[:12],
            }

    # NDJSON? (multi-line, every sampled line parses as JSON)
    lines = content.splitlines()
    if len(lines) > 1 and lines[0].lstrip()[:1] in "[{":
        sampled = 0
        ok = 0
        for ln in lines[:5]:
            ln = ln.strip()
            if not ln:
                continue
            sampled += 1
            try:
                json.loads(ln)
                ok += 1
            except (ValueError, json.JSONDecodeError):
                break
        if sampled and ok == sampled:
            return {"format": "ndjson", "lines": len(lines)}

    # Plain text fallback.
    first_line = next((ln for ln in lines if ln.strip()), "")[:200]
    return {
        "format": "text",
        "lines": len(lines),
        "first_line": first_line,
    }


def _build_meta(
    tc_id: str,
    content: str,
    tool_name: str,
    args: dict | None,
    path: str | None,
) -> dict:
    return {
        "tc_id": tc_id,
        "tool_name": tool_name or "",
        "args": args or {},
        "ts": time.time(),
        "path": path or "",
        "total_chars": len(content),
        "total_lines": len(content.splitlines()),
        "preview": _format_aware_preview(content),
    }


def _read_content(tc_id: str) -> str | None:
    """Return stored content for ``tc_id`` or None if missing."""
    p = _path_for(tc_id)
    if p is not None:
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return _memory.get(tc_id)


# ─── Public API ──────────────────────────────────────────────────────


def save(
    tc_id: str,
    content: str,
    *,
    tool_name: str = "",
    args: dict | None = None,
) -> None:
    """Persist the full tool output under ``tc_id``.

    ``tc_id`` comes from the OpenAI tool_call ID (e.g., ``call_abc123``),
    which is unique within a conversation. Empty IDs are ignored so a
    malformed tool call can't create a ``_.txt`` bucket that swallows
    subsequent outputs.

    ``tool_name`` and ``args`` are optional context fields written to
    the manifest sidecar. Old call sites that just pass ``(tc_id,
    content)`` keep working — they just produce a manifest with empty
    tool_name/args.
    """
    if not tc_id:
        return
    p = _path_for(tc_id)
    meta_p = _meta_path_for(tc_id)
    if p is not None:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            meta = _build_meta(tc_id, content, tool_name, args, str(p))
            if meta_p is not None:
                try:
                    meta_p.write_text(
                        json.dumps(meta, default=str), encoding="utf-8"
                    )
                except OSError:
                    # Manifest write is best-effort; the .txt is the
                    # source of truth, so we don't fail the save just
                    # because the sidecar didn't land.
                    pass
            _memory_meta[tc_id] = meta
            return
        except OSError:
            # Fall through to memory so the fetch path still works if
            # the disk is somehow unwritable mid-session.
            pass
    _memory[tc_id] = content
    _memory_meta[tc_id] = _build_meta(tc_id, content, tool_name, args, None)


def fetch(tc_id: str, offset: int = 0, limit: int = 4000) -> str:
    """Return a slice of the stored output, or an error string.

    Callers provide ``offset`` and ``limit`` to page through large
    outputs. We return up to ``limit`` chars so the caller can paginate
    without re-blowing the context window.
    """
    content = _read_content(tc_id)
    if content is None:
        return f"ERROR: no stored output for tool_call_id={tc_id!r}"

    total = len(content)
    if offset < 0:
        offset = 0
    if offset >= total:
        return f"[offset {offset} is past end of {total}-char output]"
    end = min(total, offset + limit)
    slice_ = content[offset:end]
    header = f"[chars {offset}:{end} of {total}]\n"
    if end < total:
        slice_ += (
            f"\n\n[...{total - end:,} more chars — "
            f"call fetch_tool_output with offset={end}]"
        )
    return header + slice_


def search(tc_id: str, pattern: str, max_matches: int = 20, context: int = 1) -> str:
    """Return lines matching ``pattern`` (regex) with context.

    ``context`` is the number of surrounding lines to include above and
    below each match. ``max_matches`` caps the number of matches we
    *render*; matches beyond the cap are counted in the trailing
    ``... and N more`` message.
    """
    content = _read_content(tc_id)
    if content is None:
        return f"ERROR: no stored output for tool_call_id={tc_id!r}"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: invalid regex {pattern!r}: {e}"

    lines = content.splitlines()
    hits: list[int] = [i for i, line in enumerate(lines) if rx.search(line)]
    if not hits:
        return f"No matches for {pattern!r} in tool_call_id={tc_id}"

    header = f"[{len(hits)} matches for {pattern!r} in {len(lines)} lines]"
    capped = hits[:max_matches]
    omitted = len(hits) - len(capped)
    match_set = set(capped)

    # Merge the (start, end) range for each capped match into non-overlapping
    # blocks so a match absorbed by a prior match's context doesn't get
    # printed twice or skipped entirely.
    blocks: list[tuple[int, int]] = []
    for idx in capped:
        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)
        if blocks and start <= blocks[-1][1]:
            prev_start, prev_end = blocks[-1]
            blocks[-1] = (prev_start, max(prev_end, end))
        else:
            blocks.append((start, end))

    out: list[str] = [header]
    for start, end in blocks:
        for i in range(start, end):
            marker = ">" if i in match_set else " "
            out.append(f"{marker} {i + 1}: {lines[i]}")
        out.append("")  # blank line between blocks

    if omitted:
        out.append(f"... and {omitted} more matches")
    return "\n".join(out).rstrip()


def list_outputs(
    tool_name: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return saved manifest entries (newest first).

    Reads from the on-disk sidecars when the disk backend is active,
    otherwise from the in-process ``_memory_meta`` mirror. ``tool_name``
    filters to a single tool's outputs; ``limit`` caps the row count.
    """
    rows: list[dict] = []
    d = _tool_outputs_dir()
    if d is not None and d.exists():
        for child in d.iterdir():
            if child.suffix != ".json":
                continue
            try:
                rows.append(json.loads(child.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    else:
        rows = list(_memory_meta.values())

    if tool_name:
        rows = [r for r in rows if r.get("tool_name") == tool_name]
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return rows[:limit]


def get_meta(tc_id: str) -> dict | None:
    """Return the manifest entry for ``tc_id``, or None if missing."""
    meta_p = _meta_path_for(tc_id)
    if meta_p is not None and meta_p.exists():
        try:
            return json.loads(meta_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return _memory_meta.get(tc_id)


def clear() -> None:
    """Drop everything from the active backend. Used by tests and agent.reset()."""
    _memory.clear()
    _memory_meta.clear()
    d = _tool_outputs_dir()
    if d is not None and d.exists():
        for child in d.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except OSError:
                    pass


def size() -> int:
    """Number of stored outputs in the active backend. For tests/debug.

    Counts only the ``.txt`` payload files so the manifest sidecars
    don't double the count.
    """
    d = _tool_outputs_dir()
    if d is not None and d.exists():
        return sum(
            1 for c in d.iterdir() if c.is_file() and c.suffix == ".txt"
        )
    return len(_memory)
