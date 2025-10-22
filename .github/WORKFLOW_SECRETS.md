# Workflow Secrets for AiDotNet CI/CD

Set these GitHub repository secrets (Settings → Secrets and variables → Actions → New repository secret):

- `CODECOV_TOKEN`: Codecov upload token
  - Used by: `.github/workflows/ci.yml` (coverage upload)

- `NUGET_API_KEY`: NuGet API key for publishing packages
  - Used by: `.github/workflows/release.yml` (publish to NuGet)

Important:
- Do NOT commit secrets to the repository.
- These secrets may be shared across multiple repos; after rollout here, update other repos to ensure consistency.
- If rotating keys, update all dependent repos’ secrets simultaneously to avoid broken pipelines.

#

