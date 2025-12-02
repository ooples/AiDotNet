## PR Title (Auto-Fixed)
**Note:** PR titles are automatically fixed to follow [Conventional Commits](https://www.conventionalcommits.org/) format for automated releases.

The workflow will intelligently detect the appropriate type based on:
- Title keywords (fix/add/implement/update/etc.)
- Files changed (docs/tests/ci/source files)
- Default to `chore:` if unsure

If the auto-detected type is incorrect, simply edit the PR title manually.

## User Story / Context
- Reference: [US-XXX] (if applicable)
- Base branch: `merge-dev2-to-master`

## Summary
- What changed and why (scoped strictly to the user story / PR intent)

## Verification
- [ ] Builds succeed (scoped to changed projects)
- [ ] Unit tests pass locally
- [ ] Code coverage >= 90% for touched code
- [ ] Codecov upload succeeded (if token configured)
- [ ] TFM verification (net46, net6.0, net8.0) passes (if packaging)
- [ ] No unresolved Copilot comments on HEAD

## Copilot Review Loop (Outcome-Based)
Record counts before/after your last push:
- Comments on HEAD BEFORE: [N]
- Comments on HEAD AFTER (60s): [M]
- Final HEAD SHA: [sha]

## Files Modified
- [ ] List files changed (must align with scope)

## Notes
- Any follow-ups, caveats, or migration details
