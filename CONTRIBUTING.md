# Contributing

## Base Branch
- Use `merge-dev2-to-master` as the working base for CI and PRs.

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

