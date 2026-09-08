# Automated Release Pipeline - Installation Instructions

## Overview

This directory contains an automated release pipeline template that provides:

- **Automated Semantic Versioning** from conventional commits
- **Changelog Generation** with categorized changes
- **GitHub Releases** with NuGet package artifacts
- **NuGet Publishing** to nuget.org
- **100% Automated** - no manual version bumps needed

## Release Workflow

The installed workflow is `.github/workflows/release-please.yml`. It maintains a rolling Release PR and publishes only after a maintainer merges that Release PR.

## Prerequisites

### Step 1: Configure NuGet Trusted Publishing

Create a trusted-publishing policy at [nuget.org](https://www.nuget.org/account/trustedpublishing) with these exact GitHub Actions values:

| Field | Value |
|-------|-------|
| Owner | `ooples` |
| Repository | `AiDotNet` |
| Workflow file | `release-please.yml` |
| Environment | Leave blank |

Enter only the workflow filename, not `.github/workflows/release-please.yml`. The policy is case-sensitive and must authorize every package published by this workflow. `NuGet/login` exchanges the job's GitHub OIDC identity for a short-lived API key; no long-lived NuGet API key is stored in GitHub.

Publishing fails closed if the policy does not match or NuGet does not issue a temporary key. See [NuGet trusted publishing](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing).

### Step 2: Verify Repository Secrets and Variables

| Name | Kind | Required | Description |
|------|------|----------|-------------|
| `AIDOTNET_BUILD_KEY` | Secret | Required for a release | Strong-name/integrity build key injected only into the unprivileged build job |
| `AUTOFIX_PAT` | Secret | Recommended | Lets release-please-created events trigger the normal protected CI workflows |
| `GITHUB_TOKEN` | Automatic | Automatic | Used by GitHub Actions; do not create it manually |
| `AIDOTNET_LICENSE_PUBLIC_KEY_JSON` | Variable or secret | Optional | Overrides the committed public license-verification key during a key rotation |
| `AIDOTNET_LICENSE_REVOCATION_JSON` | Variable or secret | Optional | Injects the signed license revocation list |

Configure repository secrets and variables under Repository Settings → Secrets and variables → Actions.

### Step 3: Review Branch Configuration

The workflow triggers on pushes to `main` and `master` branches:

```yaml
on:
  push:
    branches:
      - main
      - master
```

If your default branch has a different name, update this in the workflow file.

### Step 4: Verify the Installed Workflow

Confirm that `.github/workflows/release-please.yml` is present on the protected default branch and that GitHub Actions is enabled. The trusted-publishing policy must continue to name `release-please.yml` if the workflow is renamed.

## How It Works

### Versioning Rules

The workflow analyzes commit messages since the last git tag:

| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `BREAKING CHANGE:` or `!` | MAJOR | 0.0.5 → 1.0.0 |
| `feat:` | MINOR | 0.0.5 → 0.1.0 |
| `fix:` | MINOR | 0.0.5 → 0.1.0 |
| `refactor:` | MINOR | 0.0.5 → 0.1.0 |
| `perf:` | MINOR | 0.0.5 → 0.1.0 |
| `docs:` | MINOR | 0.0.5 → 0.1.0 |
| Other types | No release | - |

### Workflow Jobs

1. **release-please**
   - Maintains the rolling Release PR
   - Creates the version tag and GitHub release only when that PR is merged

2. **build-release**
   - Checks out the immutable release tag
   - Builds, signs, packs, and verifies all NuGet packages without OIDC permission
   - Uploads the verified packages as an immutable, one-day workflow artifact

3. **publish**
   - Downloads the exact artifact ID produced by `build-release`
   - Uses `NuGet/login` and OIDC to obtain a short-lived NuGet API key
   - Publishes the verified packages and attaches them to the GitHub release
   - Has no source checkout, build scripts, signing key, or package mutation steps

### Changelog Format

The workflow generates categorized changelogs:

```markdown
## Changes in v0.1.0

### 🚨 Breaking Changes
- Major API redesign

### ✨ Features
- Add neural network support
- Add convolutional layers

### 🐛 Bug Fixes
- Fix memory leak in training loop

### ⚡ Performance Improvements
- Optimize matrix multiplication

### ♻️ Code Refactoring
- Simplify activation functions

### 📚 Documentation
- Update API reference
```

## Testing the Workflow

Workflow syntax, action pins, permission boundaries, and artifact handoff should be validated in the pull request before merge. A normal push to `main` or `master` updates the rolling Release PR but does not publish a package.

The complete OIDC exchange can only be proven by merging a release-please Release PR because the publish jobs require `release_created == 'true'`, an immutable release tag, and a matching NuGet trusted-publishing policy. For that release run, verify:

1. `build-release` succeeds without `id-token: write`.
2. `publish` downloads the artifact ID emitted by `build-release`.
3. `NuGet/login` issues a temporary key without a repository NuGet secret.
4. Every expected package appears on nuget.org and on the GitHub release.
5. The temporary key is never printed and is unavailable to build steps.

## Conventional Commits

To use this workflow effectively, all commits must follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Examples

```bash
# Feature (MINOR bump)
git commit -m "feat: add support for LSTM layers"

# Bug fix (MINOR bump)
git commit -m "fix: correct gradient calculation in backpropagation"

# Breaking change (MAJOR bump)
git commit -m "feat!: redesign tensor API

BREAKING CHANGE: Constructor signature has changed"

# Documentation (MINOR bump)
git commit -m "docs: add API reference for neural networks"

# No release
git commit -m "chore: update dependencies"
```

See [CONVENTIONAL_COMMITS_GUIDE.md](CONVENTIONAL_COMMITS_GUIDE.md) for detailed guidance.

## Verification

After installation, verify the workflow is working:

1. **Check Workflow File**
   ```bash
   ls -la .github/workflows/release-please.yml
   ```

2. **View in GitHub**
   - Go to the repository Actions tab
   - Look for the "Release Please" workflow

3. **Verify a Normal Push**
   - Confirm a conventional commit updates the rolling Release PR without publishing

4. **Verify the Next Deliberate Release**
   - Go to Actions tab
   - Merge the reviewed release-please Release PR
   - Confirm `release-please`, `build-release`, and `publish` complete in sequence

## Troubleshooting

### Workflow Doesn't Trigger

**Problem**: Pushed to main but workflow didn't run.

**Solutions**:
- Verify `.github/workflows/release-please.yml` exists
- Check branch name matches workflow trigger (main vs master)
- Ensure GitHub Actions are enabled in repository settings

### No Release Created

**Problem**: Workflow runs but no release is created.

**Solutions**:
- Ensure commits use conventional commit format (feat:, fix:, etc.)
- Check workflow logs for "No conventional commits found"
- Verify at least one commit since last tag

### NuGet Publish Fails

**Problem**: Package not published to NuGet.

**Solutions**:
- Verify the NuGet trusted-publishing policy uses owner `ooples`, repository `AiDotNet`, workflow file `release-please.yml`, and a blank environment
- Verify the policy authorizes every package produced by the release
- Confirm the `publish` job has `id-token: write` and `NuGet/login` issued a temporary key
- Ensure package version doesn't already exist on NuGet
- Review NuGet publish logs in workflow

### TFM Verification Fails

**Problem**: A required target-framework assembly is missing from the package.

**Solutions**:
- Verify `src/AiDotNet.csproj` has:
  ```xml
  <TargetFrameworks>net10.0;net471</TargetFrameworks>
  ```
- Ensure project builds successfully for both targets locally:
  ```bash
  dotnet build src/AiDotNet.csproj -c Release
  ```

### Permission Errors

**Problem**: "push declined due to repository rule violations"

**Solutions**:
- Verify you have write access to the repository
- Check branch protection rules allow workflow modifications
- Ensure you're not trying to modify the workflow from a GitHub App

## Documentation

- [VERSIONING.md](VERSIONING.md) - Detailed versioning guide
- [CONVENTIONAL_COMMITS_GUIDE.md](CONVENTIONAL_COMMITS_GUIDE.md) - Commit message guide
- [workflows/release-please.yml](workflows/release-please.yml) - Installed release workflow

## Support

If you encounter issues:

1. Review workflow logs in the Actions tab
2. Check the troubleshooting section above
3. Verify all prerequisites are met
4. Open an issue with logs and error messages

## Maintenance

### Updating the Workflow

To update the workflow:

1. Modify `.github/workflows/release-please.yml`
2. Test changes on a feature branch first
3. Merge to main when verified

### Disabling Auto-Release

To temporarily disable automatic releases:

```yaml
on:
  workflow_dispatch:  # Only manual triggers
```

Or delete/rename `.github/workflows/release-please.yml` and update the NuGet trusted-publishing policy before re-enabling releases.

## Security Considerations

- **OIDC**: NuGet credentials are short-lived and exist only in the minimal `publish` job
- **Isolation**: Build, signing, pack, and verification steps have no `id-token: write` permission
- **Permissions**: Each job declares only the GitHub and OIDC permissions it needs
- **Dependencies**: GitHub Actions are pinned to full commit SHAs
- **Validation**: All inputs are validated before creating releases

## Benefits

After installation, you get:

- **No Manual Versioning**: Version numbers determined automatically
- **Consistent Releases**: Every release follows the same process
- **Better Communication**: Changelogs generated from commits
- **Faster Releases**: Push to main and release happens automatically
- **Audit Trail**: All releases tracked in git tags and GitHub Releases
- **Version Conflicts Handled**: `--skip-duplicate` prevents failures

## Next Steps

1. Create and verify the NuGet trusted-publishing policy described above
2. Review [CONVENTIONAL_COMMITS_GUIDE.md](CONVENTIONAL_COMMITS_GUIDE.md)
3. Update team documentation with commit message requirements
4. Validate workflow syntax and permissions in a pull request
5. Monitor the next deliberate release and confirm all expected NuGet packages and GitHub assets

Happy releasing! 🚀
