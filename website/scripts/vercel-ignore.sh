#!/usr/bin/env bash
# Vercel ignoreCommand for the aidotnet_website project.
#
# Exit codes (per Vercel):
#   0 = ignore the build (skip)
#   1 = proceed with the build
#
# Policy:
# - master / main pushes build only when website/ changed (production
#   deploys still throttle to actual website diffs — see "Why" below).
# - On preview deploys with a prior build SHA, only build when files
#   under website/ changed. The path is rooted to the repo top so the
#   filter works regardless of which cwd Vercel hands us — the previous
#   'git diff -- .' form was cwd-dependent and silently expanded to the
#   whole repo on rootDirectory misconfigurations, which is the root
#   cause of "every commit triggers a deploy" rate-limit exhaustion.
# - Without a previous SHA (first deploy of a branch / shallow clone)
#   we still build — better to deploy unnecessarily once on a fresh
#   branch than to skip the very first preview silently.
#
# Why master also filters now:
# - We were burning the Vercel deployment quota on every src/, tests/,
#   .github/ commit because master pushes unconditionally triggered a
#   full website build. The website itself doesn't depend on any of
#   those paths, so there's no production-correctness reason to
#   rebuild it for them.
#
# Lives in scripts/ rather than inlined in vercel.json because Vercel
# enforces a 256-character cap on ignoreCommand strings.

set -u

PREV="${VERCEL_GIT_PREVIOUS_SHA:-}"
CURR="${VERCEL_GIT_COMMIT_SHA:-HEAD}"

if [ -z "$PREV" ]; then
  exit 1
fi

# Resolve repo top so the path filter is cwd-independent. Vercel has
# historically run ignoreCommand from either the repo root or the
# project's rootDirectory; pinning to the top-level rev makes the
# diff scope deterministic across both.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

if git -C "$REPO_ROOT" diff --quiet "$PREV" "$CURR" -- 'website/'; then
  exit 0
else
  exit 1
fi
