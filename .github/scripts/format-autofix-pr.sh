#!/usr/bin/env bash
# Opens or updates the pull request for the formatting batch that format-autofix-batch.sh published.
# It never merges, approves or enables auto-merge.
#
# Environment (all set by the workflow):
#   GH_TOKEN, GITHUB_REPOSITORY, GITHUB_REPOSITORY_OWNER
#   FORMAT_BRANCH, FORMAT_BASE (default master), FORMAT_BATCH_MAX_FILES
#   OPEN_PR_NUMBER    number of the open pull request for FORMAT_BRANCH, or empty
#   BRANCH_SHA        commit FORMAT_BRANCH points at after the batch step
#   BATCH_FILES, TOTAL_FILES, REMAINING_FILES, SKIPPED_FILES, HELD, OBSOLETE
#   HAS_PAT           'true' when the run authenticates with AUTOFIX_PAT rather than the job token
#   AUTOFIX_REVIEWERS optional comma-separated reviewers; default is the repository owner
#
# Decisions:
#   - OBSOLETE=true (an open batch with nothing left to format): the pull request is closed, unless HELD
#     says a person has committed to its branch, in which case it is left open with a warning.
#   - A pull request is open: its title and body are refreshed. A failure to do so is a warning, not a
#     failed run: the batch itself is already published.
#   - None is open, and a pull request for this exact commit was closed without merging: nothing is opened.
#     Somebody rejected this batch; re-opening it every night would only ask them again. Listing the
#     offending path in the skip file (or any change to the batch) produces a new commit and a new request.
#   - Otherwise a pull request is created.
set -euo pipefail

: "${GITHUB_REPOSITORY:?}" "${FORMAT_BRANCH:?}" "${BRANCH_SHA:?}"
base=${FORMAT_BASE:-master}
summary=${GITHUB_STEP_SUMMARY:-/dev/null}
output=${GITHUB_OUTPUT:-/dev/null}
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

title='chore: automated dotnet format + line-ending normalization'
body="$workdir/body.md"
{
  echo 'Automated formatting maintenance PR. It is not merged automatically.'
  echo
  echo "This batch: **${BATCH_FILES}** file(s) of **${TOTAL_FILES}** that differ from the formatted tree, in path"
  echo "order. **${REMAINING_FILES}** file(s) wait for later batches; a batch holds at most"
  echo "${FORMAT_BATCH_MAX_FILES} files, and the next one is opened by the first run after this one is merged"
  echo '(nightly, or dispatch the workflow on master by hand).'
  echo
  echo '- `dotnet format whitespace . --folder`: whitespace only, from `.editorconfig`.'
  echo '- `git add --renormalize`: the `.gitattributes` EOL policy (dotnet format does not normalize line endings).'
  echo
  echo '**Merging.** Use "Squash and merge". The batch commit is unsigned and master requires signed commits; a'
  echo 'squash is signed by GitHub, a rebase merge is rejected.'
  echo
  echo '**A file that should not be formatted, or a batch that cannot go green.** Add the path to'
  echo '`.github/format-autofix-skip.txt` on master (one literal path per line). The next run rebuilds the batch'
  echo "without it. ${SKIPPED_FILES} drifted file(s) are on that list now."
  echo
  echo '**Fixing the batch by hand.** Commits pushed to this branch are kept: while this pull request is open the'
  echo 'workflow does not overwrite a branch that carries a commit it did not make.'
  echo
  if [ "${HAS_PAT:-false}" = true ]; then
    echo '**Checks.** Pushed and opened with `AUTOFIX_PAT`, so the required `CI Gate` check runs. Master requires'
    echo 'approval of the last push by someone other than the pusher: the account that owns `AUTOFIX_PAT` cannot'
    echo 'be the approving reviewer.'
  else
    echo '**Checks.** `AUTOFIX_PAT` is not configured, so this was pushed and opened with the job token and GitHub'
    echo 'starts no workflows for it: the required `CI Gate` check will not appear until the secret is set (scopes'
    echo '`repo` and `workflow`) and the batch content changes, or the pull request is closed and reopened by a person.'
  fi
} > "$body"

number=${OPEN_PR_NUMBER:-}
if [ "${OBSOLETE:-false}" = true ]; then
  # Nothing the open batch holds needs formatting any more (formatted on the base, or on the skip list).
  if [ -z "$number" ]; then
    echo "pull_number=" >> "$output"
    exit 0
  fi
  if [ "${HELD:-false}" = true ]; then
    echo "::warning title=Obsolete formatting batch kept::PR #$number has nothing left to format, but its branch carries commits this workflow did not make; it is left open for the people working on it."
    {
      echo '### Formatting batch'
      echo "PR #$number has nothing left to format but carries commits this workflow did not make; it was **not closed**."
    } >> "$summary"
  else
    gh pr close "$number" --repo "$GITHUB_REPOSITORY" \
      --comment 'Closed by the formatting workflow: none of the files in this batch needs formatting any more (they were formatted on the base branch or added to `.github/format-autofix-skip.txt`).' ||
      echo "::warning::Could not close obsolete formatting PR #$number."
    {
      echo '### Formatting batch'
      echo "PR #$number closed: nothing it held needs formatting any more (${SKIPPED_FILES} drifted file(s) on the skip list)."
    } >> "$summary"
  fi
  echo "pull_number=$number" >> "$output"
  exit 0
fi
if [ -n "$number" ]; then
  echo "Updating existing PR #$number"
  gh api -X PATCH "repos/$GITHUB_REPOSITORY/pulls/$number" -f title="$title" -F body=@"$body" --silent ||
    echo "::warning::Could not refresh the title and body of PR #$number; the batch on $FORMAT_BRANCH is published regardless."
else
  rejected=$(gh pr list --repo "$GITHUB_REPOSITORY" --head "$FORMAT_BRANCH" --base "$base" --state closed --limit 50 \
    --json number,mergedAt,headRefOid \
    --jq "[.[] | select(.mergedAt == null and .headRefOid == \"$BRANCH_SHA\")][0].number // empty")
  if [ -n "$rejected" ]; then
    echo "::notice title=Formatting batch not re-opened::PR #$rejected for this same commit was closed without merging. Add the path that blocks it to .github/format-autofix-skip.txt, or reopen #$rejected."
    {
      echo '### Formatting batch'
      echo "Not opened: PR #$rejected for commit \`$BRANCH_SHA\` was closed without merging."
      echo "${TOTAL_FILES} file(s) still differ from the formatted tree. Add the blocking path to"
      echo '`.github/format-autofix-skip.txt` on master, or reopen that pull request.'
    } >> "$summary"
    echo "pull_number=" >> "$output"
    exit 0
  fi

  echo 'Creating new PR'
  url=$(gh pr create --repo "$GITHUB_REPOSITORY" --base "$base" --head "$FORMAT_BRANCH" --title "$title" --body-file "$body")
  number=${url##*/}
  gh pr edit "$number" --repo "$GITHUB_REPOSITORY" --add-label chore --add-label autofix --add-label formatting ||
    echo "::warning::Could not apply labels to PR #$number."
  reviewers=${AUTOFIX_REVIEWERS:-${GITHUB_REPOSITORY_OWNER:-}}
  reviewers=${reviewers// /}
  if [ -n "$reviewers" ]; then
    gh pr edit "$number" --repo "$GITHUB_REPOSITORY" --add-reviewer "$reviewers" ||
      echo "::warning::Could not request review from $reviewers on PR #$number."
  fi
fi

echo "pull_number=$number" >> "$output"
{
  echo '### Formatting batch'
  echo "PR #$number: ${BATCH_FILES} file(s) of ${TOTAL_FILES} drifted; ${REMAINING_FILES} left for later batches; ${SKIPPED_FILES} on the skip list."
  if [ "${HELD:-false}" = true ]; then
    echo
    echo "The branch carries commits this workflow did not make, so it was **not refreshed**."
  fi
} >> "$summary"
