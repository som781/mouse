---
name: git-workflow
description: >
  Stage, commit, branch, rebase, and resolve merge conflicts safely.
  Use when the user asks for git operations, wants to commit, push,
  open a PR, or recover from a git mistake.
triggers:
  - keywords: [git, commit, branch, rebase, merge, push, pull, pr, pull request, stash, conflict, cherry-pick, revert]
---

# Git Workflow

Git is destructive on its sharper edges. Move slowly when state is
ambiguous. Read before you write.

## Before any git operation

1. `git status` — what's actually staged and unstaged?
2. `git log --oneline -5` — what's the recent history look like?
3. If something looks unfamiliar, **investigate before deleting**.
   Unknown branches, files, or stashes may be the user's in-progress
   work.

## Committing

- **One logical change per commit.** Don't mix a refactor with a
  bug fix.
- **Stage specific files** (`git add path/to/file`) instead of
  `git add -A` or `git add .` — those can sweep in `.env` files,
  build artifacts, or other things you didn't mean to commit.
- **Write the message in imperative mood**: "Add X", "Fix Y",
  "Refactor Z" — not "Added", "Adds", or "Adding".
- **First line ≤ 72 chars**, blank line, then a paragraph explaining
  *why* the change exists if it's not obvious.
- **Never amend a commit that's already pushed** unless you're sure
  no one else has based work on it.

## Branching

- Branch off the integration branch (`main`/`master`/`develop`),
  not off another feature branch unless you mean to.
- Name branches by intent: `fix/login-redirect`, `feat/user-export`,
  `chore/upgrade-deps`.

## Rebasing and merging

- **Rebase your feature branch onto the latest main** before opening
  a PR — fewer merge conflicts for the reviewer.
- **Merge with `--no-ff` for feature branches** if your team wants
  visible history; squash if they want a flat log. Match the team
  convention; don't switch styles.
- **Never rebase shared branches.** Rebasing rewrites history. If
  someone else has a commit on top of yours, you'll orphan their work.

## Conflicts

- Resolve conflicts by understanding what each side meant, not by
  picking one side mechanically.
- After resolving, run the tests. Then `git add` the resolved files
  and `git rebase --continue` (or `git merge --continue`).
- If you get lost mid-rebase: `git rebase --abort` returns you to
  the starting state. It's safe.

## Recovery

- `git reflog` — find every HEAD position from the last 90 days.
  Almost nothing is truly lost.
- `git reset --hard` — destructive; only run if you're certain.
- `git push --force` — overwrites the remote. Use `--force-with-lease`
  instead so you don't clobber someone else's commits.
- Never force-push to `main`/`master`. If asked, warn the user first.

## Pull requests

- Title is the headline (under 70 chars). Body has the details.
- Include: what changed, why, how to test, any risk.
- Link related issues.
- Keep PRs small. A 200-line PR gets a real review; a 2000-line PR
  gets a rubber stamp.
