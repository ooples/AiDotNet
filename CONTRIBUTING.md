# Contributing

## Base Branch
- Use `merge-dev2-to-master` as the working base for CI and PRs.

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
```
<type>(<optional scope>): <description in lowercase>
```

### Valid Types and Version Impact
- `feat:` - New feature (triggers MINOR version bump, e.g., 0.1.0 → 0.2.0)
- `fix:` - Bug fix (triggers MINOR version bump)
- `docs:` - Documentation changes (triggers MINOR version bump)
- `refactor:` - Code refactoring (triggers MINOR version bump)
- `perf:` - Performance improvement (triggers MINOR version bump)
- `test:` - Test additions/changes (no release)
- `chore:` - Build/tooling changes (no release)
- `ci:` - CI/CD changes (no release)
- `style:` - Code formatting (no release)

### Breaking Changes
Add `!` after type or `BREAKING CHANGE:` in commit body to trigger MAJOR version bump (e.g., 0.1.0 → 1.0.0):
```
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

