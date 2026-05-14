---
name: testing
description: >
  Write, run, and fix automated tests. Use when creating new tests,
  debugging test failures, increasing coverage, or setting up a test
  framework.
triggers:
  - keywords: [test, tests, testing, pytest, unittest, jest, vitest, coverage, fixture, mock, assertion]
  - file_pattern: "test_*.py"
  - file_pattern: "*_test.py"
  - file_pattern: "*.test.ts"
  - file_pattern: "*.test.tsx"
  - file_pattern: "*.test.js"
  - file_pattern: "*.spec.ts"
  - file_pattern: "*.spec.js"
---

# Testing

A good test is fast, deterministic, and tells you what broke when it
fails. Tests are the contract — write them like the next person reading
them is a stranger trying to understand the system.

## Before writing a test

1. **Read the existing tests.** Match their structure, naming, and
   fixture style. Inconsistent tests rot fastest.
2. **Find the right level.** Unit tests for logic, integration tests
   for boundaries, end-to-end tests for critical user paths. Don't
   write a slow integration test when a unit test would do.
3. **Identify the seams.** What can be injected? What needs mocking?
   What shouldn't be mocked because mocking it makes the test useless?

## Writing the test

- **One assertion per concept**, not per `assert` statement. Multiple
  asserts that all check the same property are fine; asserts checking
  unrelated things should be split into separate tests.
- **Name the test by the behavior being verified**, not the function:
  `test_returns_zero_when_list_is_empty` not `test_sum`.
- **Arrange / Act / Assert** — set up state, do the thing, check the
  outcome. Use blank lines to separate.
- **No randomness.** Seed RNGs. Freeze time. Pin fixtures.
- **No network unless that's the point.** Network tests belong in a
  separate, slow tier.
- **Assert on values, not on call counts** unless the call count is
  the thing under test.

## Fixing a failing test

1. **Read the failure message before touching code.** Most test
   failures are the test telling the truth about a broken assumption.
2. **Reproduce locally** before making any changes.
3. **Don't comment out the assertion.** Don't loosen it to make it
   pass. If the new behavior is correct, update the assertion to
   match the new contract — and explain why in the commit.

## Running tests

- Run only the failing test first: `pytest tests/test_x.py::test_y -v`
- Run the whole file once it passes
- Run the whole suite before committing
- Use `-x` to stop at the first failure when iterating

## Coverage

Coverage is a floor, not a goal. 100% coverage of trivial code is
worse than 80% coverage of the risky code. Aim coverage at:
- Error paths and edge cases
- Newly added or recently changed code
- Anything with branches the type checker can't see

Don't write tests just to bump the number.
