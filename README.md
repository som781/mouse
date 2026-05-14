# mouse

A lightweight, generic agent harness for wrapping any LLM into a
reliable, tool-using, context-aware agent. Model-agnostic
(LiteLLM-backed), MCP-native, skill-aware, with structural
primitives — not prompt coaching — for grounding, pruning, and
tool-output addressability.

Mouse is a runtime, not a framework. It ships six built-in tools,
loads everything else from MCP or skills on demand, and refuses to
encode any tool-, server-, or domain-specific behavior in harness
code.

## Docs

- **[`docs/OVERVIEW.md`](./docs/OVERVIEW.md)** — what mouse is, the
  five principles we live by, what's shipped, what we deliberately
  don't do. Start here.

## Quickstart

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .

# configure MCP servers (copy and edit)
cp mcp.json.example mcp.json

# run
python -m mouse
```

## Tests

```bash
python -m pytest
```

## License

MIT — see [`LICENSE`](./LICENSE).
