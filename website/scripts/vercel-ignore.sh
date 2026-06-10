#!/usr/bin/env bash
# Vercel ignoreCommand for the aidotnet_website project.
#
# Exit codes (per Vercel):
#   0 = ignore the build (skip)   1 = proceed with the build
#
# Goal: build ONLY when files under website/ actually changed — for both
# production (master/main) and preview (PR) deployments — so unrelated src/,
# tests/, .github/ churn doesn't burn the Vercel deployment quota.
#
# Why the previous version still rebuilt on every PR (the rate-limit bug):
#   1. It keyed off VERCEL_GIT_PREVIOUS_SHA and did `exit 1` (build) whenever it
#      was empty. That env var is EMPTY on the first deployment of every PR
#      branch, so every freshly-opened PR rebuilt unconditionally.
#   2. `git diff PREV CURR` returns 128 (fatal: bad object) when PREV is not in
#      Vercel's shallow clone. `if git diff ...; then exit 0; else exit 1; fi`
#      conflates that 128 ERROR with exit-1 "has changes", so a missing PREV
#      silently fell through to build as well.
#
# Fix: pick a RELIABLE base to diff against, then only build on a real website/
# diff. Treat "cannot determine a base" (never "diff errored") as the sole
# build-anyway fallback, so we never silently skip a genuine website change.

set -u

CURR="${VERCEL_GIT_COMMIT_SHA:-HEAD}"
REF="${VERCEL_GIT_COMMIT_REF:-}"
PREV="${VERCEL_GIT_PREVIOUS_SHA:-}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
cd "$REPO_ROOT" || exit 1

# Resolve a base commit to diff CURR against:
#   1) the previous successful deployment SHA, if it is present in this clone;
#   2) production branch: the parent commit (master advances one commit at a time);
#   3) preview branch: the merge-base with the production branch (the PR's fork
#      point), fetched shallowly — this captures the PR's WHOLE diff, so a PR that
#      never touches website/ is skipped even on its first push.
BASE=""
if [ -n "$PREV" ] && git cat-file -e "${PREV}^{commit}" 2>/dev/null; then
  BASE="$PREV"
elif [ "$REF" = "master" ] || [ "$REF" = "main" ]; then
  BASE="$(git rev-parse "${CURR}^" 2>/dev/null || true)"
else
  git fetch --no-tags --depth=200 origin master 2>/dev/null \
    || git fetch --no-tags --depth=200 origin main 2>/dev/null || true
  BASE="$(git merge-base FETCH_HEAD "$CURR" 2>/dev/null || true)"
fi

# Could not establish a base (truly unknowable history) -> build to be safe.
[ -z "$BASE" ] && exit 1

# `git diff --quiet` exits 0 (no diff) or 1 (diff). Map explicitly so a >1 error
# can never masquerade as "has changes".
git diff --quiet "$BASE" "$CURR" -- 'website/'
case "$?" in
  0) exit 0 ;;  # no website/ change -> skip the build
  1) exit 1 ;;  # website/ changed   -> build
  *) exit 1 ;;  # diff itself failed -> build (fail safe)
esac
