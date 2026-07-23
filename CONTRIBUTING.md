# Contributing

## Base Branch
- Use **`master`** as the base for all PRs (trunk-based development).
- Create short-lived feature branches from `master` (`feat/...`, `fix/...`, `docs/...`, `chore/...`, `perf/...`, `refactor/...`, `test/...`, `ci/...`, `build/...`, `audit/...`).
- Keep PRs small and focused. The semantic-release pipeline assigns the version bump from your Conventional Commits subject (see below).
- Legacy note: an older `merge-dev2-to-master` integration branch existed prior to v0.205.0; it has been retired and is not used.

## Conventional Commits (Auto-Fixed)
All PR titles are automatically converted to follow [Conventional Commits](https://www.conventionalcommits.org/) specification for automated releases.

### How It Works
When you create or update a PR, a GitHub workflow automatically:
1. Detects if your title follows conventional commits format
2. If not, intelligently determines the correct type based on:
   - Title keywords (`fix`, `add`, `implement`, `update`, etc.)
   - Files changed (documentation, tests, CI, source code)
3. Automatically updates your PR title
4. Posts a comment explaining the change

### Format
```text
<type>(<optional scope>): <description>
```

### Valid Types and Version Impact
Per audit-2026-05 finding #18, only `feat:` and breaking changes bump
MINOR/MAJOR respectively; everything else is PATCH. The earlier
"fix/refactor/perf/docs → MINOR + test/chore/ci/style → None" mapping
was a semver.org violation (a bug fix should never bump MINOR, and a
test or CI change is still a real change that the release pipeline
should record). See `.github/VERSIONING.md` for the authoritative
rules — the table here mirrors them.

- `feat:` - New feature (triggers MINOR version bump, e.g., 0.1.0 → 0.2.0)
- `fix:` - Bug fix (triggers PATCH version bump)
- `docs:` - Documentation changes (triggers PATCH version bump)
- `refactor:` - Code refactoring (triggers PATCH version bump)
- `perf:` - Performance improvement (triggers PATCH version bump)
- `test:` - Test additions/changes (triggers PATCH version bump)
- `chore:` - Build/tooling changes (triggers PATCH version bump)
- `ci:` - CI/CD changes (triggers PATCH version bump)
- `style:` - Code formatting (triggers PATCH version bump)
- `build:` - Build/packaging changes (triggers PATCH version bump)
- `revert:` - Reverts a prior commit (triggers PATCH version bump)

### Breaking Changes
Add `!` after type or `BREAKING CHANGE:` in commit body to trigger MAJOR version bump (e.g., 0.1.0 → 1.0.0):
```text
feat!: redesign public API
```

### Examples
Before auto-fix → After auto-fix:
- `Fix issue 408 in AiDotNet` → `fix: fix issue 408 in AiDotNet`
- `Implement autodiff backward passes` → `feat: implement autodiff backward passes`
- `Update outdated GitHub README file` → `docs: update outdated GitHub README file`

**Note:** If the auto-detected type is incorrect, you can manually edit the PR title. PR titles become merge commit messages, which the automated release pipeline uses to determine version bumps and generate changelogs.

## Copilot Review Loop (Mandatory)
1. Get the PR `headRefOid` (HEAD SHA).
2. Retrieve review comments and filter to those where `commit_id == headRefOid` and author matches "copilot".
3. Apply suggestions exactly (or implement an equivalent real fix).
4. Commit (no force push), wait 30–60s for re-review, re-check unresolved count.
5. Iterate until unresolved count = 0.

## CI/CD Expectations
- Multi-TFM: net46, net6.0, net8.0.
- Coverage ≥ 90% for modified code paths; upload to Codecov if `CODECOV_TOKEN` is set.
- Release workflow validates packaged TFMs before publishing.

## Scope Discipline
- Only modify files relevant to the user story or bug.
- Don’t introduce unrelated refactors.

## YAML Hygiene
- No literal `\n` in names; correct indentation; steps under `steps`.
- Shell `if` blocks must close with `fi` before next step.
- Use `${{ ... }}` expression syntax for job/step `if` where supported.

