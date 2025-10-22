# Branch Protection Guidance for AiDotNet

We recommend enabling branch protection on both `merge-dev2-to-master` (active base) and `master` (release) with these required status checks:

- CI (.NET) / Lint and Format Check
- CI (.NET) / Build
- CI (.NET) / Test (.NET 8.0.x)
- CI (.NET) / Integration Tests
- CI (.NET) / All Checks Passed
- Quality Gates (.NET) / Publish Size Analysis
- Commit Message Lint / commitlint

Also enable:
- Require pull request reviews (e.g., 1–2 approvals)
- Dismiss stale approvals on new commits (optional)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require CODEOWNERS review

Note: The exact names shown in GitHub’s UI may include the workflow/job prefixes. Use the checks as they appear on a PR to configure the protection rules.
