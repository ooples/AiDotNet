#!/usr/bin/env bash
# Self-test for format-autofix-batch.sh and format-autofix-pr.sh. Needs only git, bash and coreutils:
# every repository is a scratch one, the remote is a local bare repository and `gh` is a stub.
#
#   bash .github/scripts/test-format-autofix.sh
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
batch_script="$here/format-autofix-batch.sh"
pr_script="$here/format-autofix-pr.sh"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
export GIT_CONFIG_GLOBAL="$root/gitconfig" GIT_CONFIG_NOSYSTEM=1
git config --global user.name 'github-actions[bot]'
git config --global user.email 'github-actions[bot]@users.noreply.github.com'
git config --global core.autocrlf false
git config --global core.eol lf
git config --global init.defaultBranch master
git config --global advice.detachedHead false

cases=0
failures=0
check() { # description, condition already evaluated: $2 is 0 for pass
  cases=$((cases + 1))
  if [ "$2" -ne 0 ]; then failures=$((failures + 1)); echo "FAIL: $1"; fi
}
expect() { # description, actual, expected
  cases=$((cases + 1))
  if [ "$2" != "$3" ]; then failures=$((failures + 1)); echo "FAIL: $1: expected '$3', got '$2'"; fi
}
out() { grep "^$1=" "$root/out" | tail -1 | cut -d= -f2-; }

# A fresh world: bare remote, a `dev` clone that plays master's committers, and `work`, the checkout the
# workflow would have. Files: a/0.cs .. a/5.cs, plus names git could mistake for patterns or options.
new_world() {
  rm -rf "$root/remote.git" "$root/dev" "$root/work"
  git init -q --bare "$root/remote.git"
  git init -q "$root/dev"
  (
    cd "$root/dev" && git remote add origin "$root/remote.git"
    printf '* text=auto\n' > .gitattributes
    mkdir -p a
    for i in 1 2 3 4 5; do printf 'class C%s {}\n' "$i" > "a/$i.cs"; done
    printf 'class B {}\n' > 'a/w[1].cs'; printf 'class W {}\n' > a/w1.cs
    printf 'class S {}\n' > 'a/sp ace.cs'; printf 'class D {}\n' > 'a/-dash.cs'
    git add -A && git commit -qm init
    # a/crlf.cs is committed with CRLF in the blob, as old commits in this repository are.
    printf 'x\r\ny\r\n' > "$root/crlf.bin"
    blob=$(git hash-object -w --no-filters "$root/crlf.bin")
    git update-index --add --cacheinfo "100644,$blob,a/crlf.cs" && git commit -qm crlf
    cp "$root/crlf.bin" a/crlf.cs
    git push -q origin HEAD:master
  )
  git clone -q "$root/remote.git" "$root/work" 2>/dev/null
}
# The "formatter": every tracked .cs file except the CRLF one gets a trailing marker line, once.
format_tree() {
  (
    cd "$root/work"
    git ls-files -z '*.cs' | while IFS= read -r -d '' f; do
      [ "$f" = a/crlf.cs ] && continue
      tail -n 1 -- "$f" | grep -q '^// fmt$' || printf '// fmt\n' >> "$f"
    done
  )
}
# Bring `work` to the remote's master, as a new run's checkout does, and format it.
sync_work() {
  ( cd "$root/work" && git fetch -q origin && git checkout -q -f --detach origin/master && git clean -qfd )
  format_tree
}
run_batch() { # max-files [FORMAT_OPEN_PR]
  : > "$root/out"
  ( cd "$root/work" && FORMAT_OPEN_PR=${2:-false} GITHUB_OUTPUT="$root/out" bash "$batch_script" auto/fmt "$1" origin origin/master ) > "$root/log" 2>&1
}
branch_sha() { git --git-dir="$root/remote.git" rev-parse --verify --quiet refs/heads/auto/fmt || true; }
branch_files() { git --git-dir="$root/remote.git" diff --name-only "$(git --git-dir="$root/remote.git" merge-base master auto/fmt)" auto/fmt | tr '\n' ' '; }
dev_commit() { # message; the caller has already changed files in dev
  ( cd "$root/dev" && git pull -q origin master && git add -A && git commit -qm "$1" && git push -q origin HEAD:master )
}

# ── batch script ─────────────────────────────────────────────────────────────────────────────────────

new_world; sync_work
echo junk > "$root/work/untracked.txt"
run_batch 3; expect 'limit 3: exit code' "$?" 0
expect 'limit 3: total drift (9 formatted + 1 CRLF)' "$(out total_files)" 10
expect 'limit 3: batch size' "$(out batch_files)" 3
expect 'limit 3: pushed' "$(out pushed)" true
expect 'limit 3: files are the first three in ordinal order, names taken literally' "$(branch_files)" 'a/-dash.cs a/1.cs a/2.cs '
expect 'limit 3: branch_sha is the pushed commit' "$(out branch_sha)" "$(branch_sha)"
first=$(branch_sha)

run_batch 3
expect 'rerun: not pushed' "$(out pushed)" false
expect 'rerun: branch untouched' "$(branch_sha)" "$first"

run_batch 100
expect 'limit 100: everything in one batch' "$(out batch_files)" 10
check 'limit 100: untracked file is not committed' "$(git --git-dir="$root/remote.git" cat-file -e auto/fmt:untracked.txt 2>/dev/null; [ $? -ne 0 ]; echo $?)"
expect 'limit 100: bracket name selects only itself and its sibling by their own drift' \
  "$(branch_files)" 'a/-dash.cs a/1.cs a/2.cs a/3.cs a/4.cs a/5.cs a/crlf.cs a/sp ace.cs a/w1.cs a/w[1].cs '
expect 'limit 100: CRLF blob is normalized to LF' "$(git --git-dir="$root/remote.git" cat-file -p auto/fmt:a/crlf.cs | od -An -c | tr -d ' \n')" 'x\ny\n'

# Literal pathspecs: with only `a/w[1].cs` drifted, `a/w1.cs` must not ride along.
new_world; ( cd "$root/work" && git checkout -q -f --detach origin/master && printf '// fmt\n' >> 'a/w[1].cs' )
run_batch 100
expect 'bracket name alone: only it and the CRLF file' "$(branch_files)" 'a/crlf.cs a/w[1].cs '

# HEAD is not the base: a dispatch on a feature branch must publish nothing.
new_world
( cd "$root/work" && git checkout -q -b feature origin/master && echo f > feature1.txt && git add -A && git commit -qm feature )
format_tree
run_batch 3; expect 'HEAD not base: refused' "$?" 1
expect 'HEAD not base: nothing pushed' "$(branch_sha)" ''
check 'HEAD not base: says why' "$(grep -q 'is not origin/master' "$root/log"; echo $?)"

# Skip list: a listed path is out of the queue; the list is read from the checked-out base.
new_world
( cd "$root/dev" && mkdir -p .github && printf '# reason\r\n\r\na/1.cs\r\na/not-drifted.cs\r\n' > .github/format-autofix-skip.txt )
dev_commit 'skip a/1.cs'; sync_work
run_batch 3
expect 'skip list: one drifted path skipped' "$(out skipped_files)" 1
expect 'skip list: total excludes it' "$(out total_files)" 9
expect 'skip list: batch moves on to the next path' "$(branch_files)" 'a/-dash.cs a/2.cs a/3.cs '

# Frozen batch: with the pull request open, drift landing ahead of the batch does not reshuffle it.
new_world; sync_work; run_batch 3; first=$(branch_sha)
( cd "$root/dev" && printf 'class Z {}\n' > a/0.cs ); dev_commit 'new unformatted file sorting first'; sync_work
run_batch 3 true
expect 'frozen: reported' "$(out frozen)" true
expect 'frozen: not pushed' "$(out pushed)" false
expect 'frozen: branch untouched' "$(branch_sha)" "$first"
expect 'frozen: the new file waits' "$(out remaining_files)" 8
run_batch 3 false
expect 'no open pull request: the batch is recomputed' "$(branch_files)" 'a/-dash.cs a/0.cs a/1.cs '

# Frozen batch whose content went stale: master edited one of its files. Same files, refreshed content.
new_world; sync_work; run_batch 3; first=$(branch_sha)
( cd "$root/dev" && printf 'class C1 { int x; }\n' > a/1.cs ); dev_commit 'edit a batch file'; sync_work
run_batch 3 true
expect 'stale frozen batch: pushed' "$(out pushed)" true
expect 'stale frozen batch: same files' "$(branch_files)" 'a/-dash.cs a/1.cs a/2.cs '
check 'stale frozen batch: carries the new content' "$(git --git-dir="$root/remote.git" cat-file -p auto/fmt:a/1.cs | grep -q 'int x'; echo $?)"

# A reviewer's fixup on an open batch survives.
new_world; sync_work; run_batch 3
git clone -q --branch auto/fmt "$root/remote.git" "$root/reviewer" 2>/dev/null
( cd "$root/reviewer" && git config user.email reviewer@example.com && git config user.name reviewer &&
  printf 'class C2 {} // fixed by hand\n// fmt\n' > a/2.cs && git commit -qam 'fix: repair a/2.cs' && git push -q origin HEAD:auto/fmt )
fixup=$(branch_sha); rm -rf "$root/reviewer"
run_batch 3 true; expect 'fixup, pull request open: exit code' "$?" 0
expect 'fixup, pull request open: held' "$(out held)" true
expect 'fixup, pull request open: not pushed' "$(out pushed)" false
expect 'fixup, pull request open: reviewer commit is still the tip' "$(branch_sha)" "$fixup"
expect 'fixup, pull request open: branch_sha reports the tip' "$(out branch_sha)" "$fixup"
check 'fixup, pull request open: warning names the reason' "$(grep -q 'Formatting batch held' "$root/log"; echo $?)"
# Same author as the automation but two commits: still not the script's own work.
run_batch 3 false
expect 'fixup, no pull request: stale branch is replaced' "$(out pushed)" true

# The branch exists but cannot be fetched (a transient failure after ls-remote succeeded). With a
# reviewer's commit on it the lease would accept a push, so the run must stop instead of overwriting it.
new_world; sync_work; run_batch 3
git clone -q --branch auto/fmt "$root/remote.git" "$root/reviewer" 2>/dev/null
( cd "$root/reviewer" && git config user.email reviewer@example.com && git config user.name reviewer &&
  printf 'class C2 {} // fixed by hand\n// fmt\n' > a/2.cs && git commit -qam 'fix: repair a/2.cs' && git push -q origin HEAD:auto/fmt )
fixup=$(branch_sha); rm -rf "$root/reviewer"
mkdir -p "$root/failfetch"
printf '#!/usr/bin/env bash\n[ "$1" = fetch ] && exit 1\nexec "%s" "$@"\n' "$(command -v git)" > "$root/failfetch/git"
chmod +x "$root/failfetch/git"
: > "$root/out"
( cd "$root/work" && PATH="$root/failfetch:$PATH" FORMAT_OPEN_PR=true GITHUB_OUTPUT="$root/out" \
  bash "$batch_script" auto/fmt 3 origin origin/master ) > "$root/log" 2>&1
expect 'fetch fails: refused' "$?" 1
expect 'fetch fails: reviewer commit is still the tip' "$(branch_sha)" "$fixup"
expect 'fetch fails: nothing reported as pushed' "$(out pushed)" ''
check 'fetch fails: says why' "$(grep -q 'could not be fetched and inspected' "$root/log"; echo $?)"
run_batch 3 true
expect 'fetch recovers: reviewer commit is held, not overwritten' "$(out held)" true
expect 'fetch recovers: tip unchanged' "$(branch_sha)" "$fixup"

# Every file of an open batch goes onto the skip list: the batch is obsolete and is reported as such so
# the pull request step can close it; a branch a person committed to is reported as held.
only_bracket_drift() {
  ( cd "$root/work" && git fetch -q origin && git checkout -q -f --detach origin/master && git clean -qfd &&
    printf '// fmt\n' >> 'a/w[1].cs' )
}
new_world; only_bracket_drift; run_batch 100
expect 'obsolete setup: batch is the bracket file and the CRLF file' "$(branch_files)" 'a/crlf.cs a/w[1].cs '
batch_tip=$(branch_sha)
( cd "$root/dev" && mkdir -p .github && printf 'a/crlf.cs\na/w[1].cs\n' > .github/format-autofix-skip.txt )
dev_commit 'skip the whole batch'; only_bracket_drift
run_batch 100 true; expect 'obsolete: exit code' "$?" 0
expect 'obsolete: no changes' "$(out has_changes)" false
expect 'obsolete: reported' "$(out obsolete)" true
expect 'obsolete: automation-owned branch is not held' "$(out held)" false
expect 'obsolete: branch_sha is the batch' "$(out branch_sha)" "$batch_tip"
run_batch 100 false
expect 'no open pull request: nothing is obsolete' "$(out obsolete)" false
git clone -q --branch auto/fmt "$root/remote.git" "$root/reviewer" 2>/dev/null
( cd "$root/reviewer" && git config user.email reviewer@example.com && git config user.name reviewer &&
  printf 'class B {} // by hand\n' > 'a/w[1].cs' && git commit -qam 'fix: by hand' && git push -q origin HEAD:auto/fmt )
rm -rf "$root/reviewer"
run_batch 100 true
expect 'obsolete, reviewer commit: reported' "$(out obsolete)" true
expect 'obsolete, reviewer commit: held' "$(out held)" true

# Nothing drifted.
new_world; ( cd "$root/work" && git checkout -q -f --detach origin/master )
( cd "$root/dev" && git ls-files -z | xargs -0 git add --renormalize && git commit -qm 'normalize' && git push -q origin HEAD:master )
( cd "$root/work" && git fetch -q origin && git checkout -q -f --detach origin/master )
run_batch 100
expect 'clean tree: has_changes' "$(out has_changes)" false
expect 'clean tree: nothing pushed' "$(branch_sha)" ''

run_batch 0; expect 'max-files 0 is rejected' "$?" 2
run_batch x; expect 'max-files x is rejected' "$?" 2

# ── pull request script, with a stub gh ──────────────────────────────────────────────────────────────

mkdir -p "$root/bin"
cat > "$root/bin/gh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GH_STUB_LOG"
case "$1 $2" in
  'pr list') [ -n "${GH_STUB_REJECTED:-}" ] && echo "$GH_STUB_REJECTED"; exit 0 ;;
  'pr create') echo 'https://github.com/o/r/pull/42'; exit 0 ;;
  'pr edit') exit "${GH_STUB_EDIT_EXIT:-0}" ;;
  'api -X') exit "${GH_STUB_PATCH_EXIT:-0}" ;;
esac
exit 0
STUB
chmod +x "$root/bin/gh"
run_pr() { # extra env assignments...
  : > "$root/gh.log"; : > "$root/out"; : > "$root/summary"
  env PATH="$root/bin:$PATH" GH_STUB_LOG="$root/gh.log" GH_TOKEN=x GITHUB_REPOSITORY=o/r GITHUB_REPOSITORY_OWNER=o \
    FORMAT_BRANCH=auto/fmt FORMAT_BASE=master FORMAT_BATCH_MAX_FILES=100 BRANCH_SHA=abc123 BATCH_FILES=100 \
    TOTAL_FILES=1778 REMAINING_FILES=1678 SKIPPED_FILES=0 HELD=false HAS_PAT=true \
    GITHUB_OUTPUT="$root/out" GITHUB_STEP_SUMMARY="$root/summary" "$@" bash "$pr_script" > "$root/log" 2>&1
}
calls() { grep -c -- "$1" "$root/gh.log" || true; }

run_pr OPEN_PR_NUMBER=7; expect 'open pull request: exit code' "$?" 0
expect 'open pull request: patched through the REST API' "$(calls '^api -X PATCH repos/o/r/pulls/7 ')" 1
expect 'open pull request: none created' "$(calls '^pr create')" 0
expect 'open pull request: review is not requested again' "$(calls 'add-reviewer')" 0
expect 'open pull request: number reported' "$(out pull_number)" 7

run_pr OPEN_PR_NUMBER=7 GH_STUB_PATCH_EXIT=1; expect 'open pull request, refresh fails: run still succeeds' "$?" 0
check 'open pull request, refresh fails: warning' "$(grep -q '::warning::Could not refresh' "$root/log"; echo $?)"

run_pr OPEN_PR_NUMBER= GH_STUB_REJECTED=5; expect 'closed unmerged at this commit: exit code' "$?" 0
expect 'closed unmerged at this commit: not re-opened' "$(calls '^pr create')" 0
check 'closed unmerged at this commit: notice' "$(grep -q 'Formatting batch not re-opened::PR #5' "$root/log"; echo $?)"
check 'closed unmerged at this commit: the lookup filters on this commit and on unmerged' \
  "$(grep -q 'mergedAt == null and .headRefOid == "abc123"' "$root/gh.log"; echo $?)"
check 'closed unmerged at this commit: summary says what to do' "$(grep -q 'format-autofix-skip.txt' "$root/summary"; echo $?)"

run_pr OPEN_PR_NUMBER=; expect 'no pull request: exit code' "$?" 0
expect 'no pull request: created once' "$(calls '^pr create --repo o/r --base master --head auto/fmt ')" 1
expect 'no pull request: review requested from the owner' "$(calls 'add-reviewer o$')" 1
expect 'no pull request: number reported' "$(out pull_number)" 42

run_pr OPEN_PR_NUMBER= GH_STUB_EDIT_EXIT=1; expect 'labels and reviewers fail: run still succeeds' "$?" 0
run_pr OPEN_PR_NUMBER= HAS_PAT=false AUTOFIX_REVIEWERS='a, b'
expect 'reviewer list: spaces removed' "$(calls 'add-reviewer a,b$')" 1
run_pr OPEN_PR_NUMBER=7 OBSOLETE=true; expect 'obsolete batch: exit code' "$?" 0
expect 'obsolete batch: closed' "$(calls '^pr close 7 --repo o/r ')" 1
expect 'obsolete batch: not patched' "$(calls '^api -X PATCH')" 0
expect 'obsolete batch: none created' "$(calls '^pr create')" 0
run_pr OPEN_PR_NUMBER=7 OBSOLETE=true HELD=true; expect 'obsolete batch, reviewer commits: exit code' "$?" 0
expect 'obsolete batch, reviewer commits: not closed' "$(calls '^pr close')" 0
check 'obsolete batch, reviewer commits: warning' "$(grep -q 'Obsolete formatting batch kept' "$root/log"; echo $?)"
run_pr OPEN_PR_NUMBER= OBSOLETE=true; expect 'obsolete without a pull request: nothing created' "$(calls '^pr create')" 0

if [ "$failures" -ne 0 ]; then
  echo "$failures of $cases format auto-fix cases failed."
  exit 1
fi
echo "Format auto-fix self-test passed ($cases cases)."
