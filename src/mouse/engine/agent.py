"""Agent — the core ReAct loop."""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass

from mouse.term import C, term_width
from mouse.project import PROJECT_ROOT
from mouse.tools.registry import ToolRegistry
from mouse.tools.permissions import PermissionManager
from mouse.tools.pruner import ToolPruner
from mouse.tools import tool_output_store
from mouse.llm import LLMClient, LLMError, Usage
from mouse.skills import SkillRegistry, match_skills, render_skills_for_prompt
from mouse.skills.parser import Skill
from mouse.engine.grounding import (
    GroundingContext,
    empty_result_note,
    refusal_message,
)
from mouse.engine.events import EventLog
from mouse.memory.manager import MemoryManager
from mouse.memory.injector import build_memory_context
from mouse.sessions.manager import SessionManager


# Cap how much of a tool's raw result is kept in conversation history.
# Display still uses its own (smaller) per-turn cap. This guards against
# pathological tool outputs (e.g. paginated dumps) blowing the context
# window across many turns.
MAX_TOOL_RESULT_CHARS = 8000

# Cap on the rolling buffer of recent user messages fed to the pruner
# as half-weight scoring context. Tight on purpose: too long a window
# dilutes the current turn's signal with stale topics.
_RECENT_USER_MESSAGES_MAX = 3


def _format_items_index(preview: dict) -> str:
    """Render a json_list preview's items_index into a compact directory.

    The directory is the harness's answer to the "I only see item 1
    of 8" failure mode: when the model is shown a truncated list, it
    still needs to know that items 2..N exist and what they're
    called, so a name → id lookup ("which one is Globex?")
    resolves without re-fetching the full payload.
    """
    index = preview.get("items_index") or []
    if not index:
        return ""
    total = preview.get("len", len(index))
    lines = [f"items_index ({len(index)} of {total} shown):"]
    for i, entry in enumerate(index, start=1):
        if "id" in entry and "label" in entry:
            lines.append(
                f"  [{i}] {entry.get('id_key', 'id')}={entry['id']}  "
                f"{entry.get('label_key', 'label')}={entry['label']!r}"
            )
        elif "id" in entry:
            lines.append(f"  [{i}] {entry.get('id_key', 'id')}={entry['id']}")
        elif "label" in entry:
            lines.append(
                f"  [{i}] {entry.get('label_key', 'label')}={entry['label']!r}"
            )
        elif "value" in entry:
            lines.append(f"  [{i}] {entry['value']}")
    if preview.get("items_index_truncated"):
        lines.append(f"  ... ({total - len(index)} more items not in index)")
    return "\n".join(lines)


def _truncate_tool_result(
    result: str,
    *,
    tc_id: str = "",
    path: str | None = None,
    preview: dict | None = None,
) -> str:
    """Cap a tool result that's about to be appended to message history.

    The marker carries the locator handles the model needs to recover
    the trimmed content: the ``tool_call_id`` (for ``fetch_tool_output``
    / ``search_tool_output``), the on-disk path of the saved full
    output, and — for json_list payloads — the compact items_index
    so a name → id lookup still resolves even when the body was
    truncated mid-list.
    """
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    # Build the directory first so we can size the marker around it.
    index_block = _format_items_index(preview) if preview else ""
    locator_bits: list[str] = []
    if tc_id:
        locator_bits.append(f"tool_call_id={tc_id}")
    if path:
        locator_bits.append(f"saved at {path}")
    locator = " (" + ", ".join(locator_bits) + ")" if locator_bits else ""
    marker_tail = (
        f"\n\n[...truncated tool output to preserve context window{locator}. "
        f"Use fetch_tool_output / search_tool_output / list_tool_outputs "
        f"to access the full content."
    )
    if index_block:
        marker_tail += "\n" + index_block
    marker_tail += "]"

    # Reserve room for the (variable-length) marker so the total
    # message stays under the cap.
    head = max(500, MAX_TOOL_RESULT_CHARS - len(marker_tail) - 80)
    omitted = max(0, len(result) - head)
    sized_marker = (
        f"\n\n[...truncated {omitted:,} chars from tool output to preserve context window"
        f"{locator}. Use fetch_tool_output / search_tool_output / list_tool_outputs "
        f"to access the full content."
    )
    if index_block:
        sized_marker += "\n" + index_block
    sized_marker += "]"
    return result[:head] + sized_marker


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    model: str = "gpt-4o-mini"
    max_steps: int = 25
    max_retries: int = 2
    auto_approve: bool = False
    system_prompt: str = ""
    project_context: str = ""
    # When True, the agent prints per-step pruner drops and warns
    # when the model calls a tool that was not in the kept schema
    # list. Off by default to keep normal output clean.
    debug_pruner: bool = False
    # Fraction of the model's context window at which the agent
    # silently triggers compaction before the next LLM call. Set to
    # 0 to disable (sessions will then crash with
    # ContextWindowExceeded on long histories). 0.85 leaves enough
    # headroom for the response itself plus tool schemas.
    auto_compact_threshold: float = 0.85


@dataclass
class AgentStats:
    """Track token usage and costs."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    total_steps: int = 0
    total_errors: int = 0

    def record(self, usage: Usage):
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens

    def summary(self) -> str:
        total = self.total_input_tokens + self.total_output_tokens
        return (
            f"tokens: {total:,} "
            f"(in: {self.total_input_tokens:,} / out: {self.total_output_tokens:,}) | "
            f"steps: {self.total_steps} | tools: {self.total_tool_calls}"
        )


class Agent:
    """
    The core agent engine.

    Implements the agent loop:
      1. Send messages + tools to LLM
      2. If LLM returns tool_calls → execute via registry → append results → loop
      3. If LLM returns text → done
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: ToolRegistry,
        permissions: PermissionManager,
        llm: LLMClient | None = None,
        skills: SkillRegistry | None = None,
        pruner: ToolPruner | None = None,
        memory: MemoryManager | None = None,
        session: SessionManager | None = None,
        events: EventLog | None = None,
    ):
        self.config = config
        self.tools = tools
        self.permissions = permissions
        self.skills = skills or SkillRegistry()
        self.active_skills: list[Skill] = []
        self.pruner = pruner
        self.memory = memory
        # Optional session manager — when attached, the chat loop
        # persists every user/assistant/tool record through it so
        # /resume, /search, /progress and cross-session replay see
        # the same turns the LLM saw. Running without a session is
        # supported (tests do this); persistence is simply skipped.
        self.session = session
        # Structured event sink — every harness decision (pruner,
        # grounding, permission, tool dispatch latency, compaction)
        # flows here. Defaults to a silent no-op so tests and
        # session-less runs never touch the filesystem.
        self.events = events or EventLog.null()
        # Identity of the currently active session, set by the CLI
        # after it creates / resumes one. Used to exclude the live
        # session's own observations from the episodic preamble.
        self.session_id: str = ""
        self.session_title: str = ""
        self.stats = AgentStats()

        # LLM client can be injected for tests; otherwise built from config.
        self.llm = llm or LLMClient(
            model=config.model,
            max_retries=config.max_retries,
        )

        # Conversation memory persists across user turns
        self.messages: list[dict] = []

        # Tools invoked at any point in this session stay "sticky" —
        # the pruner will keep advertising them on later turns even
        # if the new user message doesn't score against them. Without
        # session-wide stickiness a multi-turn workflow loses access
        # to tools it already used, and the model starts inventing
        # the outputs of calls it can no longer make.
        self.sticky_tools: set[str] = set()

        # Rolling buffer of the most recent user messages, used by the
        # pruner as half-weight scoring context. Lets follow-up turns
        # like "show me the second one" inherit topic vocabulary from
        # the prior turn instead of scoring against an empty signal.
        # See ``_RECENT_USER_MESSAGES_MAX`` for the cap rationale.
        self._recent_user_messages: list[str] = []

        # Reactive grounding: everything the model has actually seen
        # (user input + tool output) plus the last error per tool.
        # Used to refuse tool calls whose id-shaped args don't appear
        # anywhere real, and to keep stale tool errors in front of the
        # model across turns.
        self.grounding = GroundingContext()

        # Build system prompt
        system_parts = []

        if config.system_prompt:
            system_parts.append(config.system_prompt)
        else:
            system_parts.append(self._default_system_prompt())

        if config.project_context:
            system_parts.append(f"\n## Project Context\n{config.project_context}")

        self.system_message = {"role": "system", "content": "\n\n".join(system_parts)}

    def _default_system_prompt(self) -> str:
        # The prompt intentionally does not enumerate tool names or
        # categorize them by origin. The ``tools=`` parameter on each
        # LLM call carries the currently-exposed schema and is the
        # single source of truth for what can be called this step.
        # Anything the prompt says beyond that risks drifting out of
        # sync with the pruner and encouraging hallucinated calls.
        # Per-tool usage hints belong in the tool's description field,
        # not here — keep this prompt agnostic of the installed tools.
        return textwrap.dedent(f"""\
            You are an expert software engineer and coding assistant.
            You have access to the user's terminal and filesystem through tools.

            ## Environment
            - Project root: {PROJECT_ROOT}
            - Tools run from the project root by default; use absolute
              paths to reach files outside it.

            ## Tools
            Every LLM call is accompanied by a list of tools you can
            invoke on that step. Use them. Do not re-implement what a
            tool already does. Never invent identifiers or other
            argument values — call a discovery tool first if you need
            one.

            ## Working Principles
            1. EXPLORE first — read files and understand context before making changes
            2. PLAN your approach — explain what you'll do before doing it
            3. EXECUTE step by step — one logical change at a time
            4. VERIFY your work — run tests or check output after changes

            ## Guidelines
            - Read existing code before modifying it
            - Keep changes minimal and focused
            - Run tests after making changes when possible
            - If something fails, diagnose the error before retrying
        """)

    def _persist(self, call: str, **kwargs) -> None:
        """Forward a record to the attached session manager if present.

        Persistence is best-effort — a broken SQLite write must never
        crash the live turn, so errors are printed dimly and swallowed.
        ``call`` is one of ``"user"``, ``"assistant"``, ``"tool"``,
        ``"tokens"``; kwargs are passed through.
        """
        if self.session is None:
            return
        try:
            if call == "user":
                self.session.record_user(kwargs["content"])
            elif call == "assistant":
                self.session.record_assistant(kwargs["content"])
            elif call == "tool":
                self.session.record_tool(
                    tool_name=kwargs["tool_name"],
                    tool_input=kwargs["tool_input"],
                    tool_output=kwargs["tool_output"],
                    tool_call_id=kwargs.get("tool_call_id", ""),
                )
            elif call == "tokens":
                self.session.update_token_count(kwargs["total"])
        except Exception as e:  # noqa: BLE001
            print(C.styled(f"  │ ⚠ session persist ({call}): {e}", C.DIM))

    def chat(self, user_message: str) -> str:
        """Process a user message through the agent loop. Returns the final response."""

        self.messages.append({"role": "user", "content": user_message})
        self.grounding.record_user_message(user_message)
        self._persist("user", content=user_message)
        self.events.emit(
            "turn.start",
            session_id=self.session_id,
            user_message_chars=len(user_message),
        )

        # Activate skills whose triggers fire for this user turn. They're
        # injected as an extra system message for the duration of this
        # chat() call only, so the context stays lean between turns.
        self.active_skills = match_skills(user_message, self.skills)

        # Build the cross-session memory preamble for this turn. The
        # hint steers the semantic lookup toward the user's message;
        # the session exclusion keeps the live session's own episodic
        # blocks from being recycled back as "recent observations".
        memory_preamble = ""
        if self.memory is not None:
            ctx = build_memory_context(
                self.memory,
                hint=user_message,
                exclude_session=self.session_id,
            )
            if not ctx.is_empty:
                memory_preamble = ctx.markdown

        turn_system = self._build_turn_system_message(memory_preamble=memory_preamble)

        # Tool pruning: compute the scoring text once (user turn + active
        # skill descriptions for broader coverage) and track which tools
        # the model actually invokes this turn — they stay "sticky" so
        # the schema doesn't vanish mid-loop.
        scoring_text = user_message
        if self.active_skills:
            scoring_text += " " + " ".join(
                f"{s.name} {s.description}" for s in self.active_skills
            )
        # Recent prior turns feed the pruner as half-weight context.
        # Built BEFORE we append the current message so the buffer
        # represents *prior* turns only.
        recent_context = " ".join(self._recent_user_messages)
        self._print_header(user_message)
        if self.active_skills:
            names = ", ".join(s.name for s in self.active_skills)
            print(C.styled(f"  ✨ Skills activated: {names}", C.MAGENTA))

        step = 0
        final_text = ""

        while step < self.config.max_steps:
            step += 1
            self.stats.total_steps += 1

            print(C.styled(f"\n  ┌─ Step {step}/{self.config.max_steps} ", C.CYAN, C.BOLD)
                  + C.styled(f"({self.llm.model})", C.DIM))

            # ── Auto-compact guard ──
            # Estimate the current history size and, if it's about to
            # blow the model's context window, run the adaptive
            # compaction pipeline *before* we call the LLM. Without this
            # a long-running session eventually sends an oversized
            # payload and gets a hard ContextWindowExceededError; with
            # it we silently reclaim space and keep going. The
            # ``replace_messages`` path resyncs grounding and enforces
            # the tool-call/tool-response pairing invariant, so the
            # compacted history is safe to feed to the next call.
            self._maybe_auto_compact(step)

            # ── Call the LLM (retries handled inside LLMClient) ──
            # Select which tools to expose on this call. With no pruner
            # configured we fall back to sending every registered tool
            # (legacy behavior).
            if self.pruner is not None:
                tool_schemas = self.pruner.select(
                    scoring_text,
                    self.tools,
                    sticky=self.sticky_tools,
                    recent_context=recent_context,
                )
                exposed_tool_names = {s["function"]["name"] for s in tool_schemas}
                all_names = set(self.tools.list_names())
                dropped = all_names - exposed_tool_names
                if step == 1 and dropped:
                    print(C.styled(
                        f"  ✂️  Pruned tools: {len(exposed_tool_names)}/"
                        f"{len(all_names)} kept",
                        C.DIM,
                    ))
                if self.config.debug_pruner and dropped:
                    print(C.styled(
                        f"  ✂️  [debug] dropped: {', '.join(sorted(dropped))}",
                        C.DIM,
                    ))
                self.events.emit(
                    "pruner.decision",
                    step=step,
                    kept=sorted(exposed_tool_names),
                    dropped=sorted(dropped),
                    sticky=sorted(self.sticky_tools),
                )
            else:
                tool_schemas = self.tools.all_schemas()
                exposed_tool_names = set(self.tools.list_names())

            try:
                response = self.llm.complete(
                    messages=[turn_system] + self.messages,
                    tools=tool_schemas,
                )
            except LLMError as e:
                self.stats.total_errors += 1
                print(C.styled(f"  │ ✗ API Error: {e}", C.RED))
                final_text = f"I encountered an API error: {e}"
                self.events.emit(
                    "llm.error",
                    step=step,
                    error=str(e),
                )
                break

            usage = Usage.from_response(response)
            self.stats.record(usage)
            assistant_message = response.choices[0].message
            self.events.emit(
                "llm.response",
                step=step,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                has_tool_calls=bool(assistant_message.tool_calls),
            )

            # ── Print any text ──
            if assistant_message.content:
                final_text = assistant_message.content
                self._print_text(assistant_message.content)

            # ── Check for tool calls ──
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                # Model is done
                self.messages.append({"role": "assistant", "content": final_text})
                break

            # ── Process tool calls ──
            self.messages.append(assistant_message.model_dump())

            for tc in tool_calls:
                self.stats.total_tool_calls += 1
                func_name = tc.function.name
                self.sticky_tools.add(func_name)  # sticky for the rest of the session
                func_args = json.loads(tc.function.arguments)
                tool = self.tools.get(func_name)
                dispatch_start = time.perf_counter()
                outcome = "ok"

                ungrounded: list[str] = []
                pruner_refused = False
                if not tool:
                    result = f"ERROR: Unknown tool '{func_name}'"
                    self._print_tool_error(func_name, result)
                    outcome = "unknown_tool"
                elif (
                    self.pruner is not None
                    and func_name not in exposed_tool_names
                ):
                    # Authoritative pruner: the tool exists in the
                    # registry but was withheld from the schema list
                    # sent to the model on this step. Refuse rather
                    # than silently dispatch — the registry is for
                    # lookup, not an end-run around pruning. The tool
                    # is already sticky above, so the retry on the
                    # next step will see it advertised.
                    result = (
                        f"HARNESS REFUSED TO DISPATCH {func_name!r}.\n"
                        "This tool exists but was not advertised to you on "
                        "this step. Re-read the tools list and pick one that "
                        "IS available, or describe the action in text and the "
                        "harness will surface the right tool on the next step."
                    )
                    pruner_refused = True
                    outcome = "pruner_refused"
                    self._print_tool_call(func_name, func_args)
                    print(C.styled(
                        "  │ 🛑 harness refused: tool not advertised this step",
                        C.YELLOW,
                    ))
                    self.events.emit(
                        "pruner.refuse",
                        tool=func_name,
                        tc_id=tc.id,
                    )
                else:
                    # ── Grounding check ──
                    # Refuse the dispatch if any id-shaped arg was
                    # never seen in prior tool output or user input.
                    # The model gets a synthetic harness result back
                    # and corrects itself on the next step — no real
                    # tool is invoked with fabricated ids.
                    ungrounded = self.grounding.check_args(func_args)
                    if ungrounded:
                        result = refusal_message(func_name, ungrounded)
                        outcome = "grounding_refused"
                        self._print_tool_call(func_name, func_args)
                        print(C.styled(
                            f"  │ 🛑 harness refused: ungrounded {', '.join(ungrounded)}",
                            C.YELLOW,
                        ))
                        self.events.emit(
                            "grounding.refuse",
                            tool=func_name,
                            tc_id=tc.id,
                            complaints=ungrounded,
                        )
                    else:
                        self._print_tool_call(func_name, func_args)

                        # ── Permission check ──
                        if self.permissions.check(tool, func_args):
                            try:
                                result = tool.handler(func_args)
                                self._print_tool_result(result)
                            except Exception as e:
                                self.stats.total_errors += 1
                                result = f"ERROR: Tool execution failed: {e}"
                                outcome = "handler_error"
                                self._print_tool_error(func_name, result)
                        else:
                            result = "DENIED: User rejected this action."
                            outcome = "permission_denied"
                            print(C.styled("  │ 🚫 Denied by user", C.RED))
                            self.events.emit(
                                "permission.deny",
                                tool=func_name,
                                tc_id=tc.id,
                            )

                # Empty-result reflection: if the real call came back
                # with nothing, append a small generic nudge so the
                # model re-checks its scope before finalizing "nothing
                # found". This does not fire on refusal or error
                # payloads — they already carry their own signal.
                if not ungrounded and not pruner_refused and tool:
                    note = empty_result_note(result)
                    if note:
                        result = result + note
                        self.events.emit(
                            "empty_result.note",
                            tool=func_name,
                            tc_id=tc.id,
                        )

                # Feed the result (real, refused, denied, or nudged)
                # back into the grounding state so subsequent calls
                # can both ground against it and see the latest error
                # per tool.
                self.grounding.record_tool_result(func_name, result)

                # Persist the FULL output before truncation so the model
                # can re-fetch it via fetch_tool_output / search_tool_output,
                # or discover it later via list_tool_outputs. The
                # tool_name/args go into the manifest sidecar so the
                # model can find this artifact by intent ("the
                # list_tasks call I made earlier") rather than only by
                # opaque tool_call_id.
                tool_output_store.save(
                    tc.id, result, tool_name=func_name, args=func_args
                )
                # Surface the tool_call_id to the grounding context so a
                # follow-up call to fetch_tool_output / search_tool_output /
                # list_tool_outputs can pass this id without being refused
                # as ungrounded. The id is something the harness itself
                # produced, so it is by construction trustworthy.
                self.grounding.record_tool_call_id(tc.id)
                saved_path = tool_output_store.path_for(tc.id)
                # Pull the format-aware preview from the manifest so the
                # truncation marker can embed the items_index — that's
                # how a truncated json_list still answers a name → id
                # lookup without re-fetching the body.
                meta = tool_output_store.get_meta(tc.id) or {}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate_tool_result(
                        result,
                        tc_id=tc.id,
                        path=saved_path,
                        preview=meta.get("preview"),
                    ),
                })
                # Persist the FULL (pre-truncation) result into the
                # session index + transcript so /resume, /progress, and
                # /search see what the LLM actually received.
                self._persist(
                    "tool",
                    tool_name=func_name,
                    tool_input=json.dumps(func_args, default=str),
                    tool_output=result,
                    tool_call_id=tc.id,
                )
                self.events.emit(
                    "tool.dispatch",
                    tool=func_name,
                    tc_id=tc.id,
                    outcome=outcome,
                    duration_ms=int((time.perf_counter() - dispatch_start) * 1000),
                    input_chars=sum(len(str(v)) for v in func_args.values()),
                    output_chars=len(result),
                )

            print(C.styled("  └──", C.CYAN))

        else:
            # Hit max steps
            print(C.styled(f"\n  ⚠️  Reached max steps ({self.config.max_steps})", C.YELLOW))
            self.events.emit("turn.max_steps_hit", steps=step)

        # Record this turn's user message in the rolling buffer so the
        # NEXT turn can use it as half-weight scoring context.
        self._recent_user_messages.append(user_message)
        if len(self._recent_user_messages) > _RECENT_USER_MESSAGES_MAX:
            del self._recent_user_messages[
                : len(self._recent_user_messages) - _RECENT_USER_MESSAGES_MAX
            ]

        # Persist the turn-ending assistant text (if any) and the
        # running token total. Recording runs exactly once per turn
        # here so LLM-error and max-steps exits still get captured.
        if final_text:
            self._persist("assistant", content=final_text)
        self._persist(
            "tokens",
            total=self.stats.total_input_tokens + self.stats.total_output_tokens,
        )
        self.events.emit(
            "turn.end",
            steps=step,
            total_tool_calls=self.stats.total_tool_calls,
            total_tokens=(
                self.stats.total_input_tokens + self.stats.total_output_tokens
            ),
        )

        self._print_footer()
        return final_text

    def reset(self):
        """Clear conversation history."""
        self.messages = []
        self.active_skills = []
        self.sticky_tools = set()
        self._recent_user_messages = []
        self.grounding.reset()
        self.stats = AgentStats()
        tool_output_store.clear()

    def _maybe_auto_compact(self, step: int) -> None:
        """Compact history in-place if it's projected to exceed the context window.

        Fires before every LLM call. The estimator is deliberately
        approximate (char/4): we'd rather compact a little early than
        hit the provider's hard ceiling mid-turn. When the threshold is
        crossed we delegate to the existing ``compact`` pipeline (stage
        1 tool-output pruning + stage 2 summarisation) and swap the
        result in through ``replace_messages`` so grounding and the
        pairing invariant are resynced in one place.

        A single ``budget.auto_compact`` event records the before/after
        token counts so the audit trail shows when the harness moved
        on its own.
        """
        threshold = getattr(self.config, "auto_compact_threshold", 0.0)
        if not threshold or threshold <= 0:
            return
        # Cheap to compute once per step; we only need the input side.
        from mouse.sessions.compaction import (
            CompactConfig,
            compact,
            estimate_tokens,
        )

        # Include the turn system message in the estimate — that's what
        # actually gets sent to the provider. Skill / memory / error
        # blocks live there and can be substantial.
        try:
            turn_system = self._build_turn_system_message()
        except Exception:
            turn_system = self.system_message
        estimate = estimate_tokens([turn_system] + self.messages)

        try:
            window = self.llm.context_window()
        except Exception:
            window = 32_000
        limit = int(window * threshold)
        if estimate <= limit:
            return

        # Pick a compaction target well below the limit so the next turn
        # has room to breathe. 40% of the window is the same ratio
        # Claude Code uses after an auto-compact.
        target = max(1_000, int(window * 0.4))
        cfg = CompactConfig(target_tokens=target)

        before = estimate
        try:
            new_msgs, report = compact(self.messages, self.llm, config=cfg)
        except Exception as e:  # noqa: BLE001 — never fail the turn over compaction
            self.events.emit(
                "budget.auto_compact",
                step=step,
                status="error",
                error=str(e),
                before_tokens=before,
            )
            return

        self.replace_messages(new_msgs)
        after = estimate_tokens([turn_system] + self.messages)
        self.events.emit(
            "budget.auto_compact",
            step=step,
            status="ok",
            before_tokens=before,
            after_tokens=after,
            limit_tokens=limit,
            window_tokens=window,
            pruned_tool_outputs=report.tool_messages_pruned,
            messages_summarized=report.messages_summarized,
        )
        print(C.styled(
            f"  ✂️  Auto-compacted: {before:,} → {after:,} tokens "
            f"(limit {limit:,} of {window:,})",
            C.MAGENTA,
        ))

    def replace_messages(self, new_messages: list[dict]) -> None:
        """Swap the conversation and resync grounding to match.

        Any operation that removes or rewrites turns (compaction,
        /resume, future truncation) must go through this method —
        ``grounding.seen`` is the substring haystack the dispatch gate
        consults, and it must not retain content the model can no
        longer see in its own context. Otherwise the gate can approve
        a fabricated identifier that happens to substring-match a
        long-dropped entry, which is the exact failure mode grounding
        exists to prevent.

        Harness-minted ``tool_call_id``\\s (in ``grounding.harness_ids``)
        are preserved across the rebuild: they address on-disk
        artifacts that outlive in-context truncation, and the model
        may legitimately re-reference them via ``fetch_tool_output``
        or ``list_tool_outputs``.
        """
        # Defensive: enforce the tool-call/tool-result pairing invariant.
        # Any producer of a message list (compaction, session replay,
        # tests) that lands mid-pair would otherwise pass orphan
        # tool-result messages straight to the next LLM call, which
        # OpenAI rejects with a schema error. Sanitising here makes
        # the guarantee local to the one write path that updates history.
        from mouse.sessions.compaction import sanitize_tool_pairs
        safe_messages = sanitize_tool_pairs(list(new_messages))

        preserved = set(self.grounding.harness_ids)
        self.grounding.reset()
        self.grounding.harness_ids.update(preserved)
        self.messages = safe_messages
        for m in self.messages:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "user":
                self.grounding.record_user_message(content)
            elif role == "tool":
                # Synthetic tool_name for rebuild — the real name isn't
                # always stored on the tool message envelope, and
                # grounding's content haystack only cares about the
                # result string. The error-tracking map keys off the
                # tool name, but a rebuilt session has no live error
                # state to preserve.
                self.grounding.record_tool_result("replayed", content)

    def _build_turn_system_message(self, *, memory_preamble: str = "") -> dict:
        """Return a system message extended with, in order, the cross-
        session memory preamble, any active skill bodies, and the
        current grounding errors block.

        The base ``self.system_message`` is unchanged between turns;
        we only extend it for the duration of a single ``chat()``
        call so the extras don't linger in context once the topic
        moves on.
        """
        parts = [self.system_message["content"]]
        if memory_preamble:
            parts.append(memory_preamble)
        if self.active_skills:
            parts.append(render_skills_for_prompt(self.active_skills))
        errors_block = self.grounding.render_errors_block()
        if errors_block:
            parts.append(errors_block)
        if len(parts) == 1:
            return self.system_message
        return {"role": "system", "content": "\n\n".join(parts)}

    # ── Pretty Printing ──

    def _print_header(self, msg: str):
        width = term_width()
        print(f"\n{'━' * width}")
        print(C.styled("  🤖 Agent", C.BOLD, C.CYAN) + f"  {msg[:width - 12]}")
        print(f"{'━' * width}")

    def _print_footer(self):
        width = term_width()
        print(C.styled(f"\n  📊 {self.stats.summary()}", C.DIM))
        print(f"{'━' * width}\n")

    def _print_text(self, text: str):
        for line in text.split("\n"):
            print(C.styled("  │ ", C.CYAN) + line)

    def _print_tool_call(self, name: str, args: dict):
        if name == "bash":
            display = args.get("command", "")
        elif name == "read_file":
            display = args.get("path", "")
        elif name == "write_file":
            display = args.get("path", "")
        elif name == "search_files":
            display = f'"{args.get("pattern", "")}" in {args.get("path", ".")}'
        elif name == "python":
            code = args.get("code", "")
            display = code.split("\n")[0][:60] + ("..." if "\n" in code else "")
        else:
            display = json.dumps(args)[:80]

        icon = {"bash": "⚡", "read_file": "📄", "write_file": "✏️ ",
                "search_files": "🔍", "list_directory": "📂", "python": "🐍"}.get(name, "🔧")

        print(C.styled(f"  │ {icon} {name}", C.GREEN, C.BOLD) + f"  {display}")

    def _print_tool_result(self, result: str):
        lines = result.split("\n")
        max_lines = 20
        for line in lines[:max_lines]:
            print(C.styled("  │   ", C.DIM) + C.styled(line[:120], C.DIM))
        if len(lines) > max_lines:
            print(C.styled(f"  │   ... ({len(lines) - max_lines} more lines)", C.DIM))

    def _print_tool_error(self, name: str, error: str):
        print(C.styled(f"  │ ✗ {name}: {error}", C.RED))


