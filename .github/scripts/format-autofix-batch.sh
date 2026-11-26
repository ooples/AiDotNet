#!/usr/bin/env bash
# Turns the formatting drift in the working tree into ONE reviewable commit of at most <max-files> files
# on top of the base branch and pushes it to <branch>. Run from the repository root, with the base branch
# checked out, after `dotnet format` has rewritten the tree.
#
#   format-autofix-batch.sh <branch> <max-files> [<push-remote>] [<base-ref>]
#
#   <push-remote>  default origin. Authentication is whatever the checkout configured for it; no token is
#                  passed on a command line or in a URL.
#   <base-ref>     default origin/master: the branch the pull request targets.
#
# Environment:
#   FORMAT_OPEN_PR    'true' when a pull request for <branch> is open. An open batch is frozen (same files,
#                     see below) and is never overwritten once somebody else has committed to it.
#   FORMAT_SKIP_FILE  default .github/format-autofix-skip.txt: literal paths, one per line, that are left
#                     out of every batch. '#' starts a comment line.
#
# What a run does:
#   1. Refuses to run unless HEAD is exactly <base-ref>. The commit is built on <base-ref> and the file
#      limit is measured against <base-ref>, so the pull request shows the batch and nothing else.
#   2. The drift is every tracked file whose formatted, EOL-normalized content differs from the base,
#      minus the skip file, in ordinal path order (so a batch is a contiguous run of folders).
#   3. The batch is the first <max-files> drifted files. While a pull request is open the batch is FROZEN
#      instead: the files the branch already holds, as long as each of them is still drifted. New drift
#      that sorts ahead of an open batch waits for a later batch instead of reshuffling it.
#   4. The branch is pushed only when its content would change. With a pull request open and a commit on
#      the branch that this script did not make (a reviewer's fixup), nothing is pushed at all: the run
#      reports held=true and leaves the branch to the people working on it.
#
# Nothing here merges anything, and a run never publishes more than one batch. The commit is built in a
# scratch index, so the checkout, the real index and HEAD are left alone.
#
# Outputs (to $GITHUB_OUTPUT when set): has_changes, total_files, batch_files, remaining_files,
# skipped_files, frozen, pushed, held, branch_sha.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <branch> <max-files> [<push-remote>] [<base-ref>]" >&2
  exit 2
fi
branch=$1
max_files=$2
push_remote=${3:-origin}
base_ref=${4:-origin/master}
open_pr=${FORMAT_OPEN_PR:-false}
skip_file=${FORMAT_SKIP_FILE:-.github/format-autofix-skip.txt}
commit_subject='chore: apply dotnet format and normalize line endings'

case "$max_files" in
  ''|*[!0-9]*) echo "max-files must be a positive integer, got '$max_files'" >&2; exit 2 ;;
esac
if [ "$max_files" -lt 1 ]; then
  echo "max-files must be a positive integer, got '$max_files'" >&2
  exit 2
fi

output=${GITHUB_OUTPUT:-/dev/null}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# File names are data, never patterns: a tracked file called `a[1].cs` must select only itself.
export GIT_LITERAL_PATHSPECS=1
# One collation for every sort and comm below, so the lists compare byte for byte.
export LC_ALL=C

count_nul() { tr -cd '\0' < "$1" | wc -c | tr -d '[:space:]'; }

# 1. The batch is committed on the base and compared with the base. A run from any other commit (a manual
#    dispatch on a feature branch) would publish that branch's own changes under the formatting pull request.
if ! base=$(git rev-parse --verify --quiet "$base_ref^{commit}"); then
  echo "::error::Base ref '$base_ref' does not resolve to a commit."
  exit 1
fi
head=$(git rev-parse HEAD)
if [ "$head" != "$base" ]; then
  echo "::error::HEAD ($head) is not $base_ref ($base). The formatting batch is only built on $base_ref."
  exit 1
fi

# 2. Everything that drifted. `--renormalize` re-applies the .gitattributes EOL policy to every tracked
#    file (dotnet format does not normalize line endings); `-u` stages the formatter's edits. Untracked
#    files are never picked up.
GIT_INDEX_FILE="$scratch/all.index" git read-tree "$base"
GIT_INDEX_FILE="$scratch/all.index" git add --renormalize .
GIT_INDEX_FILE="$scratch/all.index" git add -u .
GIT_INDEX_FILE="$scratch/all.index" git diff --cached --name-only -z "$base" | sort -z > "$scratch/drift.z"

# Paths somebody decided not to format (a file whose formatted form breaks CI, generated code, ...). The
# list is read from the checked-out base, so taking a file out of the queue is a reviewed change on master.
: > "$scratch/skip.z"
if [ -f "$skip_file" ]; then
  tr -d '\r' < "$skip_file" | grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' | tr '\n' '\0' | sort -z -u > "$scratch/skip.z" || true
fi
comm -z -23 "$scratch/drift.z" "$scratch/skip.z" > "$scratch/all.z"
comm -z -12 "$scratch/drift.z" "$scratch/skip.z" > "$scratch/skipped.z"
total=$(count_nul "$scratch/all.z")
skipped=$(count_nul "$scratch/skipped.z")

# What the branch holds now, if it exists.
remote_sha=$(git ls-remote "$push_remote" "refs/heads/$branch" | cut -f1)
remote_known=false
foreign=false
: > "$scratch/remote.z"
if [ -n "$remote_sha" ] && git fetch --quiet "$push_remote" "refs/heads/$branch" 2>/dev/null &&
   git cat-file -e "$remote_sha^{commit}" 2>/dev/null; then
  if merge_base=$(git merge-base "$base" "$remote_sha" 2>/dev/null); then
    remote_known=true
    git diff --name-only -z "$merge_base" "$remote_sha" | sort -z > "$scratch/remote.z"
    # The branch is this script's own work only if it is exactly one commit, with this script's subject and
    # the identity this run commits with. Anything else was put there by a person.
    if [ "$(git rev-list --count "$merge_base..$remote_sha")" != 1 ] ||
       [ "$(git log -1 --format=%s "$remote_sha")" != "$commit_subject" ] ||
       [ "$(git log -1 --format=%ae "$remote_sha")" != "$(git config user.email)" ]; then
      foreign=true
    fi
  else
    foreign=true
  fi
fi

if [ "$total" -eq 0 ]; then
  {
    echo "has_changes=false"
    echo "total_files=0"
    echo "batch_files=0"
    echo "remaining_files=0"
    echo "skipped_files=$skipped"
    echo "frozen=false"
    echo "pushed=false"
    echo "held=false"
    echo "branch_sha=$remote_sha"
  } >> "$output"
  echo "No formatting or line-ending drift ($skipped drifted file(s) on the skip list)."
  exit 0
fi

# 3. The batch. An open pull request keeps its file list while every file on it still needs formatting.
frozen=false
remote_count=$(count_nul "$scratch/remote.z")
if [ "$open_pr" = true ] && [ "$remote_known" = true ] && [ "$remote_count" -ge 1 ] &&
   [ "$remote_count" -le "$max_files" ] && [ -z "$(comm -z -23 "$scratch/remote.z" "$scratch/all.z" | tr '\0' '\n')" ]; then
  cp "$scratch/remote.z" "$scratch/batch.z"
  frozen=true
else
  head -z -n "$max_files" "$scratch/all.z" > "$scratch/batch.z"
fi
batch=$(count_nul "$scratch/batch.z")
remaining=$((total - batch))

GIT_INDEX_FILE="$scratch/batch.index" git read-tree "$base"
GIT_INDEX_FILE="$scratch/batch.index" git add --renormalize --pathspec-from-file="$scratch/batch.z" --pathspec-file-nul
GIT_INDEX_FILE="$scratch/batch.index" git add -u --pathspec-from-file="$scratch/batch.z" --pathspec-file-nul
tree=$(GIT_INDEX_FILE="$scratch/batch.index" git write-tree)
commit=$(git commit-tree "$tree" -p "$base" -m "$commit_subject")

# The limit is the point of this script, so it is checked on what the pull request will show: the commit
# against the merge base with the base branch, not against whatever happens to be checked out.
pr_base=$(git merge-base "$base" "$commit")
git diff --name-only -z "$pr_base" "$commit" | sort -z > "$scratch/committed.z"
committed=$(count_nul "$scratch/committed.z")
if [ "$committed" -gt "$max_files" ] || ! cmp -s "$scratch/committed.z" "$scratch/batch.z"; then
  echo "::error::Against $base_ref the batch commit changes $committed file(s), which is not the $batch selected (limit $max_files)."
  exit 1
fi

# 4. Publish only when the branch would change, and never over somebody else's commit on an open batch.
pushed=false
held=false
branch_sha=$remote_sha
up_to_date=false
if [ "$remote_known" = true ] && cmp -s "$scratch/remote.z" "$scratch/batch.z" &&
   xargs -0 -a "$scratch/batch.z" git diff --quiet "$remote_sha" "$commit" --; then
  up_to_date=true
fi

if [ "$up_to_date" = true ]; then
  echo "Branch $branch already holds this batch ($batch file(s)); not rewriting it."
elif [ "$open_pr" = true ] && [ "$foreign" = true ]; then
  held=true
  echo "::warning title=Formatting batch held::$branch has commits this workflow did not make and its pull request is open; it is not overwritten. Merge or close the pull request, or reset the branch, to let the batch refresh."
else
  if [ -n "$remote_sha" ]; then
    git push "$push_remote" "$commit:refs/heads/$branch" "--force-with-lease=refs/heads/$branch:$remote_sha"
  else
    git push "$push_remote" "$commit:refs/heads/$branch"
  fi
  pushed=true
  branch_sha=$commit
fi

{
  echo "has_changes=true"
  echo "total_files=$total"
  echo "batch_files=$batch"
  echo "remaining_files=$remaining"
  echo "skipped_files=$skipped"
  echo "frozen=$frozen"
  echo "pushed=$pushed"
  echo "held=$held"
  echo "branch_sha=$branch_sha"
} >> "$output"
echo "Drift: $total file(s) ($skipped more on the skip list). This batch: $batch (frozen: $frozen). Left for later batches: $remaining. Pushed: $pushed. Held: $held."
