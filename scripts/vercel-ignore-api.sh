#!/usr/bin/env bash
# Vercel ignoreCommand for the aidotnet-playground-api project.
#
# Exit codes (per Vercel):
#   0 = ignore the build (skip)
#   1 = proceed with the build
#
# Policy:
# - master / main pushes always build (production deploys).
# - On preview deploys with a prior build SHA available, build only when
#   files under api/ or vercel.json itself changed. This avoids spinning
#   up a Vercel build for PRs that touch only website/, src/, etc.
# - Without a previous SHA (first deploy of a branch), default to build
#   so we don't silently skip.
#
# Lives in scripts/ rather than inlined in vercel.json because Vercel
# enforces a 256-character cap on ignoreCommand strings.

set -u

if [ "${VERCEL_GIT_COMMIT_REF:-}" = master ] || [ "${VERCEL_GIT_COMMIT_REF:-}" = main ]; then
  exit 1
fi

PREV="${VERCEL_GIT_PREVIOUS_SHA:-}"
CURR="${VERCEL_GIT_COMMIT_SHA:-HEAD}"

if [ -z "$PREV" ]; then
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
if git -C "$REPO_ROOT" diff --quiet "$PREV" "$CURR" -- 'api/' 'vercel.json'; then
  exit 0
else
  exit 1
fi
