# Automated Release Pipeline - Installation Instructions

## Overview

This directory contains an automated release pipeline template that provides:

- **Automated Semantic Versioning** from conventional commits
- **Changelog Generation** with categorized changes
- **GitHub Releases** with NuGet package artifacts
- **NuGet Publishing** to nuget.org
- **100% Automated** - no manual version bumps needed

## Why a Template?

GitHub security policies prevent automated tools from directly modifying workflow files in `.github/workflows/`. This template must be manually installed by a repository maintainer with appropriate permissions.

## Installation

### Step 1: Copy the Workflow File

```bash
# From the repository root
cp .github/AUTOMATED_RELEASE_WORKFLOW.yml .github/workflows/release.yml
```

### Step 2: Verify Required Secrets

Ensure the following secrets are configured in your repository:

| Secret | Required | Description |
|--------|----------|-------------|
| `NUGET_API_KEY` | Optional | NuGet API key for publishing packages. If not set, publishing will be skipped. |
| `GITHUB_TOKEN` | Automatic | Automatically provided by GitHub Actions |

To add the NuGet API key:
1. Go to Repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `NUGET_API_KEY`
4. Value: Your NuGet API key from https://www.nuget.org/account/apikeys

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

### Step 4: Commit and Push

```bash
git add .github/workflows/release.yml
git commit -m "feat(ci): Enable automated release pipeline"
git push origin main
```

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

1. **version-and-build**
   - Parses commits and determines version
   - Creates git tag
   - Builds and tests project
   - Creates NuGet package
   - Verifies target frameworks (net462, net8.0)
   - Uploads package artifact

2. **publish-nuget**
   - Publishes to NuGet.org (if `NUGET_API_KEY` is set)
   - Handles duplicate versions gracefully with `--skip-duplicate`

3. **github-release**
   - Creates GitHub Release
   - Attaches NuGet package
   - Includes generated changelog

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

### Option 1: Push a Test Commit

```bash
git commit --allow-empty -m "feat: Test automated release pipeline"
git push origin main
```

This will trigger the workflow and create version 0.1.0 (or next MINOR bump).

### Option 2: Manual Workflow Trigger

If you want to test without creating a release, you can add workflow_dispatch trigger:

```yaml
on:
  push:
    branches:
      - main
      - master
  workflow_dispatch:  # Add this for manual testing
```

Then trigger it from Actions → Automated Release Pipeline → Run workflow.

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
   ls -la .github/workflows/release.yml
   ```

2. **View in GitHub**
   - Go to Actions tab in your repository
   - Look for "Automated Release Pipeline" workflow

3. **Test with a Commit**
   ```bash
   git commit --allow-empty -m "feat: test automated release"
   git push origin main
   ```

4. **Monitor Execution**
   - Go to Actions tab
   - Click on the running workflow
   - Watch logs for each job

## Troubleshooting

### Workflow Doesn't Trigger

**Problem**: Pushed to main but workflow didn't run.

**Solutions**:
- Verify `.github/workflows/release.yml` exists (not in .github/)
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
- Verify `NUGET_API_KEY` secret is configured
- Check API key has not expired
- Ensure package version doesn't already exist on NuGet
- Review NuGet publish logs in workflow

### TFM Verification Fails

**Problem**: "Missing net462 lib in package" error.

**Solutions**:
- Verify `src/AiDotNet.csproj` has:
  ```xml
  <TargetFrameworks>net8.0;net462</TargetFrameworks>
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
- [AUTOMATED_RELEASE_WORKFLOW.yml](AUTOMATED_RELEASE_WORKFLOW.yml) - Workflow template

## Support

If you encounter issues:

1. Review workflow logs in the Actions tab
2. Check the troubleshooting section above
3. Verify all prerequisites are met
4. Open an issue with logs and error messages

## Maintenance

### Updating the Workflow

To update the workflow:

1. Modify `.github/workflows/release.yml`
2. Test changes on a feature branch first
3. Merge to main when verified

### Disabling Auto-Release

To temporarily disable automatic releases:

```yaml
on:
  workflow_dispatch:  # Only manual triggers
```

Or delete/rename `.github/workflows/release.yml`.

## Security Considerations

- **Secrets**: Never commit `NUGET_API_KEY` or other secrets to the repository
- **Permissions**: The workflow uses minimal required permissions (contents: write)
- **Dependencies**: GitHub Actions are pinned to specific versions (e.g., `@v4`)
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

1. Install the workflow using instructions above
2. Review [CONVENTIONAL_COMMITS_GUIDE.md](CONVENTIONAL_COMMITS_GUIDE.md)
3. Update team documentation with commit message requirements
4. Test with a feature commit
5. Monitor first few releases to ensure everything works correctly

Happy releasing! 🚀
