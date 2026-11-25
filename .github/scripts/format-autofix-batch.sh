#!/usr/bin/env bash
# Turns the formatting drift in the working tree into ONE reviewable commit of at most <max-files> files
# and pushes it to <branch>. Run from the repository root after `dotnet format` has rewritten the tree.
#
#   format-autofix-batch.sh <branch> <max-files> [<push-remote>]
#
# The drift is every tracked file whose formatted, EOL-normalized content differs from HEAD. It is taken
# in ordinal path order, so a batch is a contiguous run of folders, and only the first <max-files> files
# are committed. The rest wait: once the batch is merged, the next run finds the next <max-files>.
# Nothing here merges anything, and a run never publishes more than one batch.
#
# The commit is built in a scratch index on top of HEAD, so the checkout, the real index and HEAD are
# left alone. The branch is pushed only when its content would change: an open batch that still matches
# is not rewritten every night.
#
# Outputs (to $GITHUB_OUTPUT when set): has_changes, total_files, batch_files, remaining_files, pushed.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <branch> <max-files> [<push-remote>]" >&2
  exit 2
fi
branch=$1
max_files=$2
push_remote=${3:-origin}
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

count_nul() { tr -cd '\0' < "$1" | wc -c | tr -d '[:space:]'; }

# 1. Everything that drifted. `--renormalize` re-applies the .gitattributes EOL policy to every tracked
#    file (dotnet format does not normalize line endings); `-u` stages the formatter's edits. Untracked
#    files are never picked up.
GIT_INDEX_FILE="$scratch/all.index" git read-tree HEAD
GIT_INDEX_FILE="$scratch/all.index" git add --renormalize .
GIT_INDEX_FILE="$scratch/all.index" git add -u .
GIT_INDEX_FILE="$scratch/all.index" git diff --cached --name-only -z HEAD | LC_ALL=C sort -z > "$scratch/all.z"
total=$(count_nul "$scratch/all.z")

if [ "$total" -eq 0 ]; then
  {
    echo "has_changes=false"
    echo "total_files=0"
    echo "batch_files=0"
    echo "remaining_files=0"
    echo "pushed=false"
  } >> "$output"
  echo "No formatting or line-ending drift."
  exit 0
fi

# 2. The batch: the first <max-files> drifted files, committed on top of HEAD.
head -z -n "$max_files" "$scratch/all.z" > "$scratch/batch.z"
batch=$(count_nul "$scratch/batch.z")
remaining=$((total - batch))

GIT_INDEX_FILE="$scratch/batch.index" git read-tree HEAD
GIT_INDEX_FILE="$scratch/batch.index" git add --renormalize --pathspec-from-file="$scratch/batch.z" --pathspec-file-nul
GIT_INDEX_FILE="$scratch/batch.index" git add -u --pathspec-from-file="$scratch/batch.z" --pathspec-file-nul
tree=$(GIT_INDEX_FILE="$scratch/batch.index" git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "chore: apply dotnet format and normalize line endings")

# The limit is the point of this script, so it is checked on the commit itself, not assumed.
git diff --name-only -z HEAD "$commit" | LC_ALL=C sort -z > "$scratch/committed.z"
committed=$(count_nul "$scratch/committed.z")
if [ "$committed" -gt "$max_files" ] || ! cmp -s "$scratch/committed.z" "$scratch/batch.z"; then
  echo "::error::The batch commit changes $committed file(s), which is not the $batch selected (limit $max_files)."
  exit 1
fi

# 3. Publish only when the open batch would change.
pushed=false
remote_sha=$(git ls-remote "$push_remote" "refs/heads/$branch" | cut -f1)
up_to_date=false
if [ -n "$remote_sha" ] && git fetch --quiet "$push_remote" "refs/heads/$branch" 2>/dev/null &&
   git cat-file -e "$remote_sha^{commit}" 2>/dev/null; then
  if merge_base=$(git merge-base HEAD "$remote_sha" 2>/dev/null); then
    git diff --name-only -z "$merge_base" "$remote_sha" | LC_ALL=C sort -z > "$scratch/remote.z"
    if cmp -s "$scratch/remote.z" "$scratch/batch.z" &&
       xargs -0 -a "$scratch/batch.z" git diff --quiet "$remote_sha" "$commit" --; then
      up_to_date=true
    fi
  fi
fi

if [ "$up_to_date" = true ]; then
  echo "Branch $branch already holds this batch ($batch file(s)); not rewriting it."
else
  if [ -n "$remote_sha" ]; then
    git push "$push_remote" "$commit:refs/heads/$branch" "--force-with-lease=refs/heads/$branch:$remote_sha"
  else
    git push "$push_remote" "$commit:refs/heads/$branch"
  fi
  pushed=true
fi

{
  echo "has_changes=true"
  echo "total_files=$total"
  echo "batch_files=$batch"
  echo "remaining_files=$remaining"
  echo "pushed=$pushed"
} >> "$output"
echo "Drift: $total file(s). This batch: $batch. Left for later batches: $remaining. Pushed: $pushed."
