---
name: debugging
description: >
  Diagnose bugs, crashes, unexpected behavior, and failing builds.
  Use when something is broken and the cause is not yet known.
triggers:
  - keywords: [bug, debug, error, crash, broken, fails, failing, doesn't work, isn't working, why is, traceback, stack trace, exception]
---

# Debugging

The bug is in the code, not the universe. Your job is to narrow the
search space until the cause is obvious. Don't guess — observe.

## The first five minutes

1. **Read the error message in full.** Don't skim. The line number
   and the exception class are usually enough to point at the file.
2. **Read the stack trace from the bottom up.** The bottom frame is
   where execution stopped; the top frame is the entry point.
3. **Reproduce it deterministically.** A bug you can't reproduce on
   demand is a bug you can't fix on demand.
4. **Note what you already know.** Write down: the symptom, the
   trigger, what you've already ruled out. This prevents re-doing
   work mid-debug.

## Narrowing the search

Use **bisection**, not guessing.

- **Bisect in time**: `git bisect` between a known-good commit and
  the broken one.
- **Bisect in space**: comment out half the input, half the config,
  half the test. Does the bug still happen? Move to the half that
  reproduces it.
- **Bisect in the call stack**: add a print/log at the boundary
  between two layers. Did the bad data come from above or originate
  here?

## Reading vs. running

- **Read the code first** if the function is small enough to fit in
  your head.
- **Run with logs** if the function is too big or the state is too
  complex to simulate mentally.
- **Use a debugger** if you need to inspect runtime state at a
  specific point. Set a breakpoint, don't sprinkle prints.

## Common traps

- **It worked yesterday.** Something changed. Find what — `git diff`,
  dependency updates, config changes, environment variables.
- **It works on my machine.** Compare environments: Python version,
  OS, env vars, file system case sensitivity, locale.
- **It only happens sometimes.** Look for: race conditions, time
  zones, ordering of unordered data (dict iteration, set lookup),
  uninitialized memory, network flakes.
- **The error is in line N.** Sometimes the *symptom* is at line N
  but the *cause* is far upstream — bad data was stored 10 minutes
  earlier.
- **The fix made it worse.** Revert immediately. Understand the
  cause before trying again.

## Once you find it

- **Verify the cause** by reverting the fix and confirming the bug
  comes back. Otherwise you might be fixing a symptom.
- **Add a test** that fails without your fix and passes with it.
  This is how the bug stays dead.
- **Look for siblings.** If this bug exists, similar bugs probably
  exist nearby. Grep for the same pattern.
- **Write a one-line root cause** in the commit message so the next
  person — possibly future you — doesn't have to re-derive it.

## When you're stuck

- **Explain the bug to a rubber duck** (or to the user). Forcing the
  problem into words often surfaces the wrong assumption.
- **Take a break.** Stuck-debugging burns hours that 10 minutes away
  would save.
- **Ask for help with: the symptom, the reproduction steps, what
  you've ruled out.** Not just "it's broken".
