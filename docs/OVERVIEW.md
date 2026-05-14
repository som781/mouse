# Mouse — Overview

A brief statement of what mouse is, what it isn't, and the design
discipline we work under.

## What mouse is

Mouse is a **lightweight, generic agent harness** for wrapping any
LLM into a reliable, tool-using, context-aware agent. It is a runtime,
not a framework. The goal is for any business to self-host and extend
it without writing harness code for their particular tool ecosystem.

> "Vercel removed 80% of their agent's tools and got better results.
> Fewer tools meant fewer steps, fewer tokens, faster responses,
> higher success." — The harness wins by doing less, not more.

## Core principles we live by

1. **The harness is generic.** No tool, server, domain, or scenario
   appears in harness code. If a fix requires naming a specific tool
   or topic, it belongs in a skill or system prompt, not in the
   engine.

2. **Structural over behavioral.** We add scoring, scoping, and
   addressability primitives — not English regex hacks or model
   coaching. If a change reads like "tell the model to try harder,"
   we throw it out.

3. **The harness moves, not the model.** Every harness component
   should be rippable. When a new model makes a feature unnecessary,
   remove it. When a model weakness appears, add a structural
   component to compensate.

4. **Minimal by default, extensible by design.** Six built-in
   tools. Everything else comes from MCP or skills, loaded on demand.

5. **Token-budget aware.** Every component that injects tokens
   declares its budget. The pruner, the manifest system, and the
   grounding context all exist to keep the model's working set small
   without losing reach.

## What's shipped

Working primitives in the current codebase. Each is generic — no
scenario coupling.

- **Tool registry + permissions** — ask/safe/deny + session-wide
  always-allow.
- **Dynamic tool pruner** — keyword scoring with two-tier
  exact/substring matching, MCP group unlock, session-wide sticky,
  half-weight recent-turn context for follow-up turns.
- **MCP integration** — multi-server, stdio/http/sse, per-server
  isolation, background event loop.
- **Skills engine** — three-tier discovery, trigger matching,
  metadata-only at startup, content loaded on demand.
- **Memory layers** — episodic, semantic, preferences, with a
  per-turn injector.
- **Tool-output manifests** — sidecar JSON next to every saved tool
  body, format-aware preview, items_index for json_list payloads,
  `list_tool_outputs` builtin, locator handles (`tc_id`, `path`,
  `items_index`) embedded in truncation markers. The "context
  firewall" pattern.
- **Reactive grounding context** — substring check on id-shaped args,
  recency-windowed haystack, sticky per-tool error tracking, refusal
  with synthetic tool result instead of letting fabricated ids
  through. Harness-minted `tool_call_id`s are held in a separate
  buffer so they survive compaction and resume without polluting the
  content haystack.
- **`replace_messages` primitive** — any operation that rewrites turns
  (compaction, `/resume`, future truncation) routes through one method
  that resyncs the grounding substring haystack to the visible
  messages, so the dispatch gate can't approve a fabricated id that
  happens to substring-match a long-dropped entry.
- **Structural empty-result note** — fires on payload *shape*
  (scalars-all-null, wrapped-empty-list, any-list-anywhere-empty), not
  on prose. No keyword lists, no English regex.
- **Structured event log** — one append-only JSONL per session
  (`sessions/<id>/events.jsonl`) recording every harness decision:
  pruner picks, grounding refusals, permission denies, tool dispatch
  latency and outcome, compaction. Free-form schema — no event
  registry, no validation — so post-hoc analysis ("every grounding
  refusal across N sessions") is a one-liner. Silent no-op when no
  session is attached.
- **Shell-composition permission guard** — bash commands containing
  `|`, `>`, `<`, `;`, `&`, `` ` ``, `$(`, `$((` always route to the
  ASK prompt regardless of the first token. Blocks pipe-outs,
  redirects, and command substitution from slipping under the
  safe-prefix allowlist.
- **MCP reconnection + configurable timeout** — each connection
  tracks `healthy`, flips to `False` on a transport-level exception
  (closed resource / broken pipe / connection reset), and lazily
  rebuilds the session on the next dispatch. Per-call timeout is
  read from server config (`call_timeout`), default 60 s; a timeout
  cancels the in-flight coroutine and marks the session unhealthy so
  a single hung server doesn't persist across calls.
- **Session storage** — SQLite index + JSONL transcripts, FTS5
  search, resume/fork. Each session directory co-locates tool
  outputs, the event log, and future per-session caches under one
  `sessions/<id>/` root.

## What we deliberately do not do

The discipline that keeps the harness honest:

- **No English give-up regexes.** If the harness needs to detect a
  failure mode, it does so structurally (empty result shape, error
  classification), not by phrase-matching the model's prose.
- **No prompt nudges from harness code.** Coaching the model belongs
  in skills or the system prompt, not in `engine/`.
- **No domain-specific scoring boosts.** The pruner is generic
  English tokenization with stopwords. Smarter selectors are welcome
  (embeddings, LLM routing) — domain hardcoding is not.
- **No silent identifier invention.** The grounding context refuses
  any id-shaped argument that doesn't appear in something the model
  was actually shown. Generic substring check, no schema knowledge.
- **No "always reads tool output back" fixation.** Saved artifacts
  exist to be *addressable*, not just re-read into context. The model
  decides when to fetch (or process them programmatically via
  `python` / `bash`) — the harness just makes them findable.

## Reading map

- [`../src/mouse/`](../src/mouse/) — the code. Each module has a
  module-level docstring describing its responsibility.
