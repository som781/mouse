---
name: documentation
description: >
  Write or improve READMEs, docstrings, code comments, API docs, and
  changelogs. Use when the user asks to document, explain, or describe
  code, or to improve existing documentation.
triggers:
  - keywords: [document, docs, documentation, readme, docstring, comment, explain, describe, changelog, api docs]
  - file_pattern: "README*"
  - file_pattern: "CHANGELOG*"
  - file_pattern: "*.rst"
---

# Documentation

Documentation is for the reader who doesn't have your context. The
goal is to let them succeed as fast as possible — not to show how
much work the project required.

## What to write

Different surfaces serve different readers:

| Surface       | Reader                | Answers                              |
|---------------|-----------------------|--------------------------------------|
| README        | First-time visitor    | What is this? Should I use it? How?  |
| Quick start   | Trying it now         | What do I run to see it work?        |
| API reference | Building against it   | What does this function do exactly?  |
| Tutorial      | Learning the concepts | How do I use it for a real task?     |
| Changelog     | Existing user         | What changed, what broke, what's new?|
| Docstring     | Code reader           | What does this thing do, briefly?    |
| Inline comment| Code reader           | Why this non-obvious choice?         |

Write the surface the reader actually needs. Don't pad a README with
contributor guidelines that belong in CONTRIBUTING.md.

## How to write a README

```
# Project Name

One sentence: what this is and who it's for.

## Quick start
   The shortest possible path from zero to working output.

## What it does
   2-3 paragraphs. Concrete examples beat abstract descriptions.

## Installing
   Exact commands. Show the prerequisite versions.

## Usage
   The most common task, in full.

## Configuration / API
   Link to the reference docs, don't duplicate them.

## License
```

## How to write a docstring

- **One-line summary** in imperative mood: "Return the user's full name."
- **Blank line**, then a paragraph if the behavior is non-obvious.
- **Document arguments and return value** only when the types or
  constraints aren't already clear from the signature.
- **Document exceptions** the caller can reasonably catch.
- **Show an example** for anything with non-trivial usage.
- **Don't restate the function name.** "Returns the result" for a
  function called `get_result` is wasted space.

## How to write a code comment

Comments explain **why**, not **what**. The code already says what.

Good comments:
- Document a non-obvious choice ("we use ISO format here because the
  upstream API rejects RFC 3339 with timezone offsets")
- Warn about a constraint that isn't visible locally ("must be called
  before init() — see ticket #1247")
- Mark a known limitation ("only handles UTF-8; #1502 tracks the
  fix")

Bad comments:
- Restating what the line does
- Out-of-date claims contradicting the code
- TODO without an owner or context

## How to write a changelog

Group by version. Within a version, group by type:

- **Added** — new features
- **Changed** — behavior changes that aren't bug fixes
- **Fixed** — bug fixes
- **Removed** — deleted features
- **Deprecated** — features marked for removal
- **Security** — security-relevant fixes

Each entry: one line, past tense, user-facing language. Link to PRs
or issues for details.

## What NOT to write

- Aspirational documentation describing features that don't exist
- "This module does X" comments at the top of a file named X.py
- Duplicated information that will drift out of sync
- Marketing copy in technical reference docs
