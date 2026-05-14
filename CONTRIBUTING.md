# Contributing

Thanks for considering a contribution to mouse.

## Development setup

Requires Python 3.10 or newer.

```bash
# create and activate a virtual env
python -m venv .venv
source .venv/bin/activate

# install runtime + editable package
pip install -r requirements.txt -e .

# install dev dependencies (pytest, etc.)
pip install -r requirements-dev.txt
```

## Running the test suite

```bash
python -m pytest
```

The suite is fast (well under a minute). Please run it before opening
a pull request and make sure it passes.

## Style and scope

- **Keep the harness generic.** No tool, server, or domain-specific
  behavior in harness code. If a fix requires naming a specific tool
  or topic, it belongs in a skill or system prompt, not in the
  engine. See [`docs/OVERVIEW.md`](./docs/OVERVIEW.md) for the full
  set of principles.
- **Prefer structural over behavioral fixes.** Scoring, scoping, and
  addressability primitives over English regex or prompt nudges.
- **Add or update tests** alongside any code change. Tests live in
  `tests/` and mirror the `src/mouse/` layout.

## Reporting issues

Please open a GitHub issue with a minimal reproduction (a short
script or command line) and the observed vs. expected behavior.
