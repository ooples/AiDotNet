# Automated Semantic Versioning Guide

## Overview

AiDotNet uses **fully automated semantic versioning** based on conventional commit messages. When code is merged to `main` or `master`, the release pipeline automatically:

1. Analyzes commit messages since the last release
2. Determines the appropriate version bump
3. Creates a new version tag
4. Generates a changelog
5. Publishes to NuGet
6. Creates a GitHub Release

**No manual version bumps are needed!** The version is 100% determined by your commit messages.

## Version Bump Rules

The versioning system follows these rules based on conventional commit types:

### MAJOR Version Bump (X.0.0)

Triggered by **BREAKING CHANGE** in commit messages:

```
feat: Add new feature

BREAKING CHANGE: This changes the public API
```

Example: `0.0.5` → `1.0.0`

### MINOR Version Bump (x.Y.0)

Triggered by any of these commit types:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `docs:` - Documentation updates

Examples:
```
feat: Add neural network support
fix: Correct matrix multiplication
refactor: Simplify vector operations
perf: Optimize batch processing
docs: Update API documentation
```

Example: `0.0.5` → `0.1.0`

### PATCH Version Bump (x.y.Z)

Triggered by any of the following Conventional Commits prefixes when no `feat:` or breaking change is present in the release window:

- `fix:` — Bug fix
- `perf:` — Performance improvement
- `refactor:` — Code restructuring without behavior change
- `docs:` — Documentation only
- `chore:` — Maintenance, dependency updates, tooling
- `style:` — Formatting / whitespace
- `test:` — Test additions or adjustments
- `ci:` — CI/CD configuration
- `build:` — Build system, packaging
- `revert:` — Reverting a prior commit

Example: `0.205.0` → `0.205.1`

Fallback patterns (non-conventional capitalised PR titles): `Fix`, `Update`, `Improve`, `Enhance`, `Resolve`, `Patch`, `Correct`, `Repair` also map to PATCH.

> **Versioning history note.** Releases prior to 0.205.0 mapped fix/refactor/perf/docs commits to MINOR bumps, which violated [semver.org](https://semver.org). Audit-2026-05 finding #18 fixed this. Pinning ranges (`>= 0.204.0, < 0.205.0`) against pre-0.205 releases is unsafe because those MINOR bumps may have contained only PATCH-class changes — consumers should pin to an exact version through 0.204.x.

### No Version Bump

If no conventional commits are found since the last release, the workflow will:
- Skip the release process
- Not create a new version tag
- Not publish to NuGet
- Not create a GitHub Release

## Commit Message Format

Use the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Commit Types

| Type | Description | Version Bump |
|------|-------------|--------------|
| `feat` | New feature | MINOR |
| `fix` | Bug fix | MINOR |
| `refactor` | Code refactoring | MINOR |
| `perf` | Performance improvement | MINOR |
| `docs` | Documentation changes | MINOR |
| `test` | Test updates | None |
| `chore` | Build/tooling changes | None |
| `style` | Code style changes | None |
| `ci` | CI/CD changes | None |

### Optional Scope

You can add a scope to provide more context:

```
feat(neural-net): Add backpropagation support
fix(matrix): Correct transpose operation
docs(readme): Add installation instructions
```

## Examples

### Example 1: Feature Release

**Commits since last tag (v0.0.5):**
```
feat: Add support for convolutional layers
fix: Correct activation function gradient
docs: Update API reference
```

**Result:**
- New version: `v0.1.0`
- Changelog includes all three commits
- Package published to NuGet
- GitHub Release created

### Example 2: Breaking Change Release

**Commits since last tag (v0.1.0):**
```
feat: Redesign tensor API

BREAKING CHANGE: Tensor constructor signature has changed
```

**Result:**
- New version: `v1.0.0`
- Changelog highlights breaking change
- Package published to NuGet
- GitHub Release created

### Example 3: No Conventional Commits

**Commits since last tag (v1.0.0):**
```
Update README
Fix typo
Merge branch 'dev'
```

**Result:**
- No version bump
- No release created
- Workflow completes successfully

## Changelog Generation

The release pipeline automatically generates a changelog organized by commit type:

```markdown
## Changes in v0.1.0

### 🚨 Breaking Changes
- Redesigned tensor API

### ✨ Features
- Add support for convolutional layers
- Add dropout layer implementation

### 🐛 Bug Fixes
- Correct activation function gradient
- Fix memory leak in batch processing

### ⚡ Performance Improvements
- Optimize matrix multiplication

### ♻️ Code Refactoring
- Simplify layer initialization

### 📚 Documentation
- Update API reference
- Add tutorial examples
```

## Workflow Details

### Trigger

The release workflow triggers automatically on:
- Pushes to `main` branch
- Pushes to `master` branch

### Jobs

1. **version-and-build**
   - Analyzes commits and determines version
   - Creates git tag
   - Builds project
   - Runs tests
   - Creates NuGet package
   - Verifies target frameworks (net462, net8.0)

2. **publish-nuget** (conditional)
   - Runs if `NUGET_API_KEY` secret is configured
   - Publishes package to nuget.org
   - Handles version conflicts gracefully

3. **github-release**
   - Creates GitHub Release
   - Attaches NuGet package as artifact
   - Includes generated changelog

## Configuration

### Required Secrets

| Secret | Description |
|--------|-------------|
| `NUGET_API_KEY` | NuGet API key for publishing packages (optional) |
| `GITHUB_TOKEN` | Automatically provided by GitHub Actions |

### Permissions

The workflow requires these permissions:
- `contents: write` - Create tags and releases
- `issues: write` - Comment on related issues
- `pull-requests: write` - Comment on related PRs
- `id-token: write` - OIDC token generation

## Target Frameworks

The workflow verifies that NuGet packages contain these target frameworks:
- `net462` (.NET Framework 4.6.2)
- `net8.0` (.NET 8.0)

## Best Practices

### DO ✅

- Use conventional commit messages for all changes
- Include scope for better changelog organization: `feat(layer): Add batch normalization`
- Write clear, descriptive commit messages
- Use imperative mood: "Add feature" not "Added feature"
- Reference issues in commit footer: `Closes #123`

### DON'T ❌

- Don't manually edit version in `AiDotNet.csproj`
- Don't create tags manually
- Don't use non-conventional commit messages for releases
- Don't skip CI checks

## Troubleshooting

### Release Not Triggered

**Problem:** Pushed to main but no release was created.

**Solution:** Check that commits use conventional format:
```bash
git log --oneline -10
```

Look for `feat:`, `fix:`, etc. prefixes.

### Version Conflict on NuGet

**Problem:** Package version already exists on NuGet.

**Solution:** The workflow uses `--skip-duplicate` flag. The job will succeed with a warning. Ensure commits are properly tagged.

### TFM Verification Failed

**Problem:** Package doesn't contain required target frameworks.

**Solution:** Check `AiDotNet.csproj` has:
```xml
<TargetFrameworks>net8.0;net462</TargetFrameworks>
```

## Manual Version Override (Emergency)

If you need to manually create a release:

1. Create and push a tag:
```bash
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

2. The workflow will use this tag as the baseline for future releases.

## Current Version

Check the current version:
- NuGet: https://www.nuget.org/packages/AiDotNet/
- GitHub Releases: https://github.com/ooples/AiDotNet/releases
- Git Tags: `git describe --tags --abbrev=0`

## Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
