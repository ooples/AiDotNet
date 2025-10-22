# Branch Protection (merge-dev2-to-master)

- Protect `merge-dev2-to-master` branch.
- Require pull request before merging.
- Require up-to-date with base before merging.
- Require status checks to pass:
  - CI (.NET)
  - Build
  - Quality Gates (if enabled)
- Require code owner review (via `.github/CODEOWNERS`).
- Disallow force pushes and deletions.

Note: Configure these in GitHub Settings → Branches after the first clean CI run so the exact check names are visible.
