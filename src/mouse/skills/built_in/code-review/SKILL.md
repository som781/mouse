---
name: code-review
description: >
  Review code changes for correctness, clarity, and risk before they ship.
  Use when the user asks to review a diff, audit a PR, check a file, or
  evaluate a change.
triggers:
  - keywords: [review, audit, pr, diff, check this, look at, evaluate]
  - file_pattern: "*.diff"
  - file_pattern: "*.patch"
---

# Code Review

Your job is to find real problems, not to re-write the code in your own
style. A good review is short, specific, and actionable.

## How to read a change

1. **Read the surrounding code first.** Most review mistakes come from
   judging a snippet without understanding what it's plugged into.
2. **Build a mental model of what the change is trying to do.** Then
   check whether the code actually does that.
3. **Look at the diff twice** — once for correctness, once for risk.

## What to look for (in priority order)

1. **Bugs.** Off-by-ones, null/empty handling, wrong sign, swapped
   arguments, race conditions, resource leaks, error swallowing.
2. **Security.** Injection (SQL, shell, HTML), unvalidated input,
   secrets in code, broken auth/authz, unsafe deserialization.
3. **Data integrity.** Migrations that drop columns, queries that
   could return too many rows, transactions that don't roll back.
4. **API contracts.** Breaking changes, public function renames,
   removed fields, changed response shapes.
5. **Performance cliffs.** N+1 queries, unbounded loops, sync calls
   in hot paths, missing indexes for new queries.
6. **Tests.** Does the change have tests? Do existing tests still
   exercise the changed code path?
7. **Clarity.** Confusing names, dead code, comments that contradict
   the code, magic numbers that should be constants.

## What NOT to flag

- Style nits the formatter would fix anyway
- "I would have written this differently" without a concrete reason
- Speculative future-proofing the change doesn't need
- Refactors of code that wasn't touched in the diff

## How to deliver findings

For each issue, write:
- **Where**: file path + line number
- **What**: one sentence describing the problem
- **Why it matters**: the concrete consequence
- **Suggested fix**: a specific change, not a vague direction

Group findings by severity: **blocking**, **should-fix**, **nit**.
If there are no blocking issues, say so explicitly.
