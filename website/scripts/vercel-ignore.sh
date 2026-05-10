#!/usr/bin/env bash
# Vercel ignoreCommand for the aidotnet_website project.
#
# Exit codes (per Vercel):
#   0 = ignore the build (skip)
#   1 = proceed with the build
#
# Policy:
# - master / main pushes always build (production deploys).
# - On preview deploys with a prior build SHA, only build when files
#   inside the project root (website/) changed. Vercel runs this from
#   the website/ directory (rootDirectory in the Vercel project config),
#   so 'git diff -- .' restricts the diff to that subtree.
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

if git diff --quiet "$PREV" "$CURR" -- .; then
  exit 0
else
  exit 1
fi
