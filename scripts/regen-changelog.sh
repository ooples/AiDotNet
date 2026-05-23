#!/usr/bin/env bash
# regen-changelog.sh — backfill / refresh CHANGELOG.md from `gh release list`.
#
# audit-2026-05 finding #15 remediation: prior CHANGELOG had a literal
# "[Previous changelog entries would appear here]" placeholder with an
# "[Unreleased] - 2025-12-18" header that was 5+ months stale against
# v0.204.0. This script regenerates the full release history from GitHub's
# release records, preserving the keepachangelog header.
#
# Usage:
#   scripts/regen-changelog.sh [--limit N]
#
# Requires:
#   - gh (GitHub CLI) authenticated against ooples/AiDotNet
#   - Run from the repo root
#
# Writes:
#   CHANGELOG.md (overwrites; commit the result)

set -euo pipefail

LIMIT="${1:-300}"
if [ "$LIMIT" = "--limit" ]; then
    LIMIT="${2:-300}"
fi

REPO="${GITHUB_REPOSITORY:-ooples/AiDotNet}"

cd "$(dirname "$0")/.."

cat > CHANGELOG.md <<HEADER
# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning note.** Releases prior to 0.205.0 mapped fix/perf/refactor/docs
> commits to MINOR bumps instead of PATCH (a semver.org violation tracked as
> audit-2026-05 finding #18). Starting at 0.205.0 the bump rules follow
> Conventional Commits properly — see `.github/VERSIONING.md`. Consumers
> pinning ranges across pre-0.205 releases should pin to exact versions.

> **History backfill.** Entries below were regenerated from \`gh release list\`
> by \`scripts/regen-changelog.sh\` for audit-2026-05 finding #15. Per-release
> summaries are the GitHub release title + commit count + date; for the
> per-commit detail, see the corresponding GitHub Release page linked from
> each version.

---

HEADER

echo "Generating CHANGELOG entries from gh release list --limit $LIMIT ..." >&2

# Stream releases from gh — title, tag, publishedAt — JSON-formatted so we
# parse robustly with python. Use a repo-relative path so git-bash on Windows
# (where /tmp/ is not visible to native python) and POSIX shells both work.
TMPJSON="./.changelog-tmp-releases.json"
trap 'rm -f "$TMPJSON"' EXIT

gh release list --limit "$LIMIT" --repo "$REPO" \
    --json tagName,name,publishedAt,isDraft,isPrerelease \
    > "$TMPJSON"

RELEASE_COUNT=$(python3 -c "import json; print(len(json.load(open('$TMPJSON'))))")
echo "Fetched $RELEASE_COUNT releases from $REPO." >&2

python3 - "$TMPJSON" "$REPO" <<'PY' >> CHANGELOG.md
import json, sys

with open(sys.argv[1]) as f:
    releases = json.load(f)

# Use the same repo slug the gh CLI was called with so links route
# correctly when GITHUB_REPOSITORY / REPO is overridden (forks,
# downstream mirrors). Previously hardcoded "ooples/AiDotNet" produced
# broken release links any time the workflow ran outside that repo.
repo = sys.argv[2]

# Sort newest-first by publishedAt (gh already does this but make it explicit)
releases.sort(key=lambda r: r.get("publishedAt") or "", reverse=True)

for r in releases:
    if r.get("isDraft"):
        continue
    tag = r.get("tagName") or "(untagged)"
    title = r.get("name") or tag
    published = (r.get("publishedAt") or "")[:10]
    prerelease = " (pre-release)" if r.get("isPrerelease") else ""
    # GitHub renders [TAG] anchors automatically when in the form
    # `## [vX.Y.Z]` so consumers can deep-link.
    print(f"## [{tag}] - {published}{prerelease}")
    print()
    if title and title != tag:
        # Some releases use the title to summarize. Show it as a one-line
        # summary; per-commit detail lives on the GitHub release page.
        print(f"_{title}_")
        print()
    print(f"See https://github.com/{repo}/releases/tag/{tag}")
    print()
PY

echo "CHANGELOG.md regenerated. Inspect, then commit." >&2
