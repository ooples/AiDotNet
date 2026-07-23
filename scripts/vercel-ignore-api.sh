#!/usr/bin/env bash
# Vercel ignoreCommand for the aidotnet-playground-api project.
#
# Exit codes (per Vercel):
#   0 = ignore the build (skip)   1 = proceed with the build
#
# Goal: build ONLY when files under api/ (or vercel.json) actually changed — for
# both production (master/main) and preview (PR) deployments. The playground API
# does not depend on website/, src/, tests/, .github/, etc., so it must not
# redeploy for those — that churn is what exhausts the Vercel deploy quota.
#
# Why the previous version still rebuilt constantly (the rate-limit bug):
#   1. master/main pushes did `exit 1` UNCONDITIONALLY, so every merge to master
#      (every src/tests/docs commit) redeployed the API even though api/ was
#      untouched.
#   2. Preview deploys keyed off VERCEL_GIT_PREVIOUS_SHA and did `exit 1` (build)
#      whenever it was empty — which it is on the FIRST deploy of every PR branch.
#   3. `git diff PREV CURR` returns 128 (fatal: bad object) when PREV is missing
#      from Vercel's shallow clone; the old `if/else` treated that error the same
#      as exit-1 "has changes", so a missing PREV also fell through to build.
#
# Fix: pick a RELIABLE base to diff against, then only build on a real api/ diff.
# Treat "cannot determine a base" (never "diff errored") as the sole build-anyway
# fallback, so we never silently skip a genuine api change.

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
#      point), fetched shallowly — captures the PR's WHOLE diff, so a PR that never
#      touches api/ is skipped even on its first push.
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
git diff --quiet "$BASE" "$CURR" -- 'api/' 'vercel.json'
case "$?" in
  0) exit 0 ;;  # no api/ or vercel.json change -> skip the build
  1) exit 1 ;;  # api/ or vercel.json changed   -> build
  *) exit 1 ;;  # diff itself failed            -> build (fail safe)
esac
