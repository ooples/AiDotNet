# CI Auto-Fix Bot Setup (Recommended)

## Overview

Some of our CI workflows automatically rewrite commits (for example, fixing commit messages to match our Conventional Commits rules). Repository rules such as **“Commits must have verified signatures”** require the automation to create **GPG-signed** commits, and GitHub limitations mean the automation should usually push using a **PAT** (not `GITHUB_TOKEN`) so downstream checks re-run.

This guide describes the safest setup:
- A **dedicated GitHub user** (bot) for CI automation
- A **dedicated GPG key** owned by that bot user (not a human’s personal key)
- Repository secrets used by the workflows

## 1) Create a dedicated GitHub bot user

Create a new GitHub account (example):
- Username: `AiDotNetAutofixBot`
- Email: a shared/team mailbox you control (example: `autofix-bot@yourdomain.com`)

Then add it to the org/repo:
- Add as a collaborator (minimum needed permissions; typically `Write`)
- If you use branch protection/rules, ensure this bot is allowed to push to PR branches as required by your rules

## 2) Add a PAT for the bot (to trigger CI correctly)

Why: pushes made with `GITHUB_TOKEN` often do **not** trigger other workflows/checks. A PAT avoids that limitation.

On the bot account:
- GitHub → Settings → Developer settings → Personal access tokens
- Create a classic PAT (or fine-grained PAT if your org prefers) with:
  - `repo` (needed to push and comment)
  - `workflow` (needed so the push can trigger other workflows)

Store it as a repo secret:
- `Settings → Secrets and variables → Actions → New repository secret`
- Name: `AUTOFIX_PAT`
- Value: the token

## 3) Create a dedicated GPG key for the bot

### Windows (Gpg4win)
1. Install Gpg4win (if not already installed).
2. Open PowerShell and run:
   - `gpg --quick-generate-key "AiDotNet Autofix Bot <autofix-bot@yourdomain.com>" default default 0`
   - `gpg --list-secret-keys --keyid-format=long`
3. Export the keys:
   - `gpg --armor --export "AiDotNet Autofix Bot" > autofix-public.key`
   - `gpg --armor --export-secret-keys "AiDotNet Autofix Bot" > autofix-private.key`

### Linux/macOS
1. Generate:
   - `gpg --quick-generate-key "AiDotNet Autofix Bot <autofix-bot@yourdomain.com>" default default 0`
2. Export:
   - `gpg --armor --export "AiDotNet Autofix Bot" > autofix-public.key`
   - `gpg --armor --export-secret-keys "AiDotNet Autofix Bot" > autofix-private.key`

## 4) Upload the bot’s public key to GitHub

On the **bot** GitHub account:
- Settings → “SSH and GPG keys” → “New GPG key”
- Paste the contents of `autofix-public.key`

This makes commits signed by that key show up as “Verified”.

## 5) Add required repo secrets for signed auto-fix commits

In the repository:
- Settings → Secrets and variables → Actions → New repository secret

Add:
- `AUTOFIX_GPG_PRIVATE_KEY`: paste the contents of `autofix-private.key`
- `AUTOFIX_GPG_PASSPHRASE`: the passphrase for the key (leave empty if you created a key without one)

## 6) Verify the setup

1. Create a test PR with an intentionally non-compliant commit message (for example: `deps: Bump ...` with sentence-case subject).
2. Confirm:
   - `Commit Message Lint` fails on the PR.
   - `Fix Commit Messages` runs, rewrites the commit(s), and pushes back to the PR branch.
   - `Commit Message Lint` re-runs on the updated commit and passes.
   - The rewritten commits show “Verified” in GitHub.

## Notes / Security

- Do not reuse a human’s personal signing key for CI automation.
- Keep `AUTOFIX_PAT` and `AUTOFIX_GPG_PRIVATE_KEY` restricted to repo secrets (never checked into git).
- Our auto-fix workflows should be guarded to avoid operating on forks; keep those guards in place.

