"""Grounding context — a tiny reactive sidecar that stops the agent
from calling tools with identifiers it never actually saw.

A common failure mode of LLM tool use is inventing plausible-looking
argument values instead of discovering them. The model happily passes
identifier strings it has never been shown, and the underlying tool
either errors, silently returns nothing, or — worst case — acts on
the wrong object. The harness should make this structurally hard,
without knowing anything about any particular tool.

This module maintains two reactive facts for the running session:

* :attr:`GroundingContext.seen` — every string the model has been
  shown: user messages and tool outputs. (The model's own assistant
  messages are intentionally excluded: grounding against your own
  hallucinations is not grounding.)
* :attr:`GroundingContext.last_errors` — the most recent error text
  observed per tool name. These are surfaced into the turn system
  message so a stale error stops aging out of salience.

Before dispatching a tool call the agent calls
:meth:`GroundingContext.check_args`. For any string argument whose
*key name* looks like an identifier (``*_id``, ``*_ids``, ``uuid``,
``slug``, etc. — see ``_ID_KEY_SUFFIXES``), the value must appear as
a substring of something in ``seen``. If it doesn't, the harness
refuses the dispatch and feeds a synthetic tool result back to the
model explaining what to do next. The check is intentionally coarse:
it ignores non-id keys entirely, so natural-language params like
``query``, ``message``, ``status`` slip through untouched.

Nothing in this module is aware of any specific tool, schema,
protocol, or domain. It treats tool names as opaque strings and
argument dicts as JSON. That is the whole point — it's a harness
mechanism, not a tool integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# Argument keys that look like identifiers. If the model passes a
# string under one of these, the value has to already appear in
# something it actually saw — otherwise it's fabricated.
_ID_KEY_SUFFIXES = ("_id", "_ids", "uid", "uids", "uuid", "gid", "gids", "slug", "slugs")
_ID_KEY_EXACT = frozenset({"id", "gid", "uuid", "slug", "topic", "org", "session_id"})

# Values shorter than this are not worth checking — they're trivially
# substrings of almost anything, and genuine ids are rarely this small.
_MIN_CHECKED_VALUE = 4

# Keep the seen-text and error history bounded so a long session
# doesn't grow an arbitrary in-memory log.
_MAX_SEEN_ENTRIES = 200
_MAX_ERROR_SNIPPET = 400

# Substring grounding only consults the most recent N seen entries.
# Older context stays in ``seen`` (it may still be useful for other
# purposes), but the substring haystack is narrowed for two reasons:
#   1. False-positive risk grows with haystack size — a long-dead
#      hex string from 80 turns ago can accidentally substring-match
#      a fabricated identifier the model just invented.
#   2. The model itself can no longer "see" entries that have aged
#      out of its own context, so grounding against them is
#      grounding against something the model can't actually point
#      back to. Recency is the right semantic.
_GROUNDING_WINDOW = 80


def _looks_like_id_key(key: str) -> bool:
    k = key.lower()
    if k in _ID_KEY_EXACT:
        return True
    return any(k.endswith(suf) for suf in _ID_KEY_SUFFIXES)


def _looks_like_error(text: str) -> bool:
    """Heuristic: is this tool result an error?

    We only scan the first 200 chars so a legitimate result containing
    the word "error" deep inside doesn't poison the error slot."""
    low = text[:200].lower().lstrip()
    return (
        low.startswith("error")
        or '"error"' in low
        or "error calling tool" in low
        or "access denied" in low
        or low.startswith("denied")
    )


@dataclass
class GroundingContext:
    """Per-session reactive grounding state.

    Held by the :class:`Agent` for the life of the session and
    cleared by :meth:`Agent.reset`. No disk persistence — grounding
    is *current-conversation* truth. Cross-session identity is the
    job of the memory layer, not this module.
    """

    seen: list[str] = field(default_factory=list)
    last_errors: dict[str, str] = field(default_factory=dict)
    # Harness-minted tool_call_ids live in their own set so they never
    # enter the content haystack used by :meth:`check_args`. A long
    # session can produce hundreds of opaque UUID fragments; mixing
    # them into ``seen`` would dilute substring lookups for genuine
    # content-bearing ids (the exact failure mode grounding exists to
    # catch). Consulted separately at check time for recovery-tool
    # args like ``tool_call_id``.
    harness_ids: set[str] = field(default_factory=set)

    # ── mutation ────────────────────────────────────────────────

    def record_user_message(self, text: str) -> None:
        if text and text.strip():
            self.seen.append(text)
            self._trim()

    def record_tool_result(self, tool_name: str, result: str) -> None:
        if not result:
            return
        self.seen.append(result)
        self._trim()
        if _looks_like_error(result):
            self.last_errors[tool_name] = result[:_MAX_ERROR_SNIPPET].strip()
        else:
            # A successful call supersedes the stale error from the
            # same tool — otherwise a long-dead error keeps shouting
            # at the model forever.
            self.last_errors.pop(tool_name, None)

    def record_tool_call_id(self, tc_id: str) -> None:
        """Track a harness-issued ``tool_call_id`` so subsequent calls
        to meta-recovery tools (``fetch_tool_output``,
        ``search_tool_output``, ``list_tool_outputs``) can ground
        their ``tool_call_id`` argument.

        The id lands in :attr:`harness_ids`, NOT ``seen``. Mixing
        harness-minted UUIDs into the content haystack would dilute
        substring lookups for real content ids over a long session,
        since opaque UUID fragments coexist with actual tool output
        inside the substring window.
        """
        if tc_id and tc_id.strip():
            self.harness_ids.add(tc_id.strip())

    def reset(self) -> None:
        self.seen.clear()
        self.last_errors.clear()
        self.harness_ids.clear()

    def _trim(self) -> None:
        if len(self.seen) > _MAX_SEEN_ENTRIES:
            del self.seen[: len(self.seen) - _MAX_SEEN_ENTRIES]

    # ── queries ─────────────────────────────────────────────────

    def check_args(self, args: dict) -> list[str]:
        """Return a list of ``"key=value"`` complaints for ungrounded
        identifier args. Empty list means the call is safe to dispatch.

        Only id-shaped *keys* are checked. Short values (< 4 chars) are
        skipped because they trivially match anywhere. List values are
        checked element-wise. ``None`` and non-scalar values are
        skipped. Integers are coerced to their string form so an
        integer row id (common in REST APIs) is still grounded.
        """
        if not args:
            return []
        # Only the recent slice of ``seen`` participates in the
        # substring haystack — see ``_GROUNDING_WINDOW`` for rationale.
        recent = self.seen[-_GROUNDING_WINDOW:]
        haystack = "\n".join(recent)
        complaints: list[str] = []
        for key, value in args.items():
            if not _looks_like_id_key(key):
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for v in values:
                if isinstance(v, bool) or v is None:
                    # bool is a subclass of int — exclude explicitly so
                    # a ``{"id": True}`` doesn't get stringified to "True"
                    # and accidentally look grounded.
                    continue
                if isinstance(v, str):
                    checked = v.strip()
                elif isinstance(v, int):
                    checked = str(v)
                else:
                    continue
                if len(checked) < _MIN_CHECKED_VALUE:
                    continue
                if checked in haystack:
                    continue
                # Harness-minted ids (e.g. tool_call_ids from the
                # provider envelope) live in a separate set so they
                # don't dilute the content haystack — check there too.
                if checked in self.harness_ids:
                    continue
                complaints.append(f"{key}={checked!r}")
        return complaints

    def render_errors_block(self) -> str:
        """Small markdown block listing recent tool errors, or ``""``.

        Injected into the turn system message so an error observed on
        one user turn is still visible on the next, instead of ageing
        out inside the raw tool-message stream.
        """
        if not self.last_errors:
            return ""
        lines = ["## Recent tool errors (harness-tracked)"]
        for name, err in self.last_errors.items():
            first_line = err.split("\n", 1)[0][:200]
            lines.append(f"- `{name}`: {first_line}")
        return "\n".join(lines)


_EMPTY_LITERALS = frozenset({"", "{}", "[]", "null", "(no output)", "none"})


def _walk_lists(val: Any):
    """Yield every list found anywhere inside a parsed JSON value."""
    if isinstance(val, list):
        yield val
        for item in val:
            yield from _walk_lists(item)
    elif isinstance(val, dict):
        for v in val.values():
            yield from _walk_lists(v)


def _walk_scalars(val: Any):
    """Yield every non-container leaf inside a parsed JSON value."""
    if isinstance(val, list):
        for item in val:
            yield from _walk_scalars(item)
    elif isinstance(val, dict):
        for item in val.values():
            yield from _walk_scalars(item)
    else:
        yield val


def _json_looks_empty(val: Any) -> bool:
    """Does a parsed JSON value carry zero data rows?

    Structural, not regex-based. The rule:

    * If any list appears anywhere in the structure, the result is
      "empty" iff EVERY such list is empty. This catches the common
      paginated shapes: ``{"items":[]}``, ``{"data":[],"meta":{...}}``,
      ``{"results":[],"next_cursor":null}``, ``{"data":{"items":[]}}``.
    * If no lists appear anywhere, the result is "empty" iff every
      scalar leaf is ``None`` (e.g. ``{"results": null}``). A dict
      with a populated scalar (``{"id":"abc"}``, ``{"status":"ok"}``)
      is a real record and is NOT treated as empty.

    Scalar metadata siblings to an empty list (``"next_cursor": "abc"``)
    do not disqualify the shape from being empty — the *list* is the
    data-bearing field, and its emptiness is the verdict.
    """
    if val is None:
        return True
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        if not val:
            return True
        lists = list(_walk_lists(val))
        if lists:
            return all(len(lst) == 0 for lst in lists)
        # No lists anywhere — rely on scalar content.
        return all(s is None for s in _walk_scalars(val))
    return False


def empty_result_note(result: str) -> str:
    """Return a harness nudge if ``result`` looks structurally empty.

    Empty tool results are where the model most confidently goes wrong:
    the server says "nothing matched" and the model declares success
    without re-checking whether it actually pointed its query at the
    right scope (org, filter, identifier). This note is a generic
    counter-pressure — not a command to retry, just a reminder to
    verify the discovery chain before finalizing.
    """
    stripped = (result or "").strip()
    if stripped.lower() in _EMPTY_LITERALS:
        return _EMPTY_NOTE
    if stripped and stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return ""
        if _json_looks_empty(parsed):
            return _EMPTY_NOTE
    return ""


_EMPTY_NOTE = (
    "\n\n[harness note: this tool returned a structurally empty result. "
    "Before finalizing 'nothing found' as your answer, confirm your "
    "discovery steps actually targeted the scope the user asked about "
    "(e.g. correct org, filter, identifier). Empty does not always mean "
    "absent — it often means a filter didn't match.]"
)


def refusal_message(tool_name: str, complaints: list[str]) -> str:
    """Format the synthetic tool result returned when dispatch is refused."""
    bullets = "\n".join(f"  - {c}" for c in complaints)
    return (
        f"HARNESS REFUSED TO DISPATCH {tool_name!r}.\n"
        "The following arguments were not found in any prior tool output "
        "or user message, which usually means they were invented:\n"
        f"{bullets}\n\n"
        "Before retrying, call a discovery tool (list/search/get) to "
        "obtain the real identifier, then reuse the exact value from "
        "its response. If the value is a legitimate new input, ask the "
        "user to provide it explicitly."
    )
