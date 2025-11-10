# Conventional Commits Guide for AiDotNet

## Overview

AiDotNet uses [Conventional Commits](https://www.conventionalcommits.org/) to automatically determine version numbers and generate changelogs. This guide explains how to write commit messages that work with our automated release pipeline.

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type (Required)

The type determines how the version number is bumped:

- **feat**: A new feature (MINOR version bump: 1.2.0 → 1.3.0)
- **fix**: A bug fix (PATCH version bump: 1.2.0 → 1.2.1)
- **perf**: Performance improvement (PATCH version bump)
- **refactor**: Code refactoring without changing functionality (no version bump)
- **docs**: Documentation changes (no version bump)
- **style**: Code style changes (formatting, semicolons, etc.) (no version bump)
- **test**: Adding or updating tests (no version bump)
- **build**: Build system changes (no version bump)
- **ci**: CI/CD pipeline changes (no version bump)
- **chore**: Other changes that don't modify src or test files (no version bump)

### Breaking Changes (MAJOR version bump)

To indicate a breaking change, add `!` after the type or add `BREAKING CHANGE:` in the footer:

```
feat!: completely redesign the API

BREAKING CHANGE: The old API methods have been removed.
```

This triggers a MAJOR version bump: 1.2.0 → 2.0.0

### Scope (Optional)

The scope provides context about what part of the codebase is affected:

```
feat(regression): add support for polynomial regression
fix(neural-networks): resolve gradient explosion issue
perf(matrix): optimize matrix multiplication for large matrices
```

Common scopes in AiDotNet:
- `linear-algebra`
- `statistics`
- `neural-networks`
- `regression`
- `activation-functions`
- `optimizers`
- `rag`
- `embeddings`
- `meta-learning`
- `benchmarks`
- `tests`

### Subject (Required)

- Use imperative mood: "add" not "added" or "adds"
- Don't capitalize the first letter
- No period at the end
- Keep it concise (50 characters or less)

### Body (Optional)

- Explain **why** the change was made, not **what** was changed
- Include motivation for the change and contrast with previous behavior
- Wrap at 72 characters

### Footer (Optional)

- Reference issues: `Closes #123` or `Fixes #456`
- Document breaking changes: `BREAKING CHANGE: <description>`

## Examples

### Feature Addition (Minor Version Bump)

```
feat: add GELU activation function

Implement the Gaussian Error Linear Unit (GELU) activation function
for use in transformer models. GELU has shown better performance than
ReLU in many NLP tasks.

Closes #234
```

### Bug Fix (Patch Version Bump)

```
fix(matrix): resolve determinant calculation for singular matrices

The determinant calculation was incorrectly throwing an exception for
singular matrices instead of returning zero.

Fixes #456
```

### Breaking Change (Major Version Bump)

```
feat(api)!: redesign PredictionModelBuilder interface

BREAKING CHANGE: The Configure methods now accept only interface types
instead of concrete implementations. This provides better dependency
injection support but requires updating existing code.

Migration guide:
- Old: builder.ConfigureOptimizer(new AdamOptimizer())
- New: var optimizer = new AdamOptimizer(); builder.ConfigureOptimizer(optimizer)

Closes #789
```

### Performance Improvement (Patch Version Bump)

```
perf(vector): optimize dot product calculation

Replace naive loop with SIMD instructions for 3x speedup on large
vectors. Benchmarks show improvement from 1500ns to 500ns for vectors
of size 10000.
```

### Multiple Types in One Commit

If a commit includes multiple types of changes, use the most significant type:

- Breaking change? Use the type with `!`
- Feature and fixes? Use `feat`
- Multiple fixes? Use `fix`

Better yet: make separate commits for different types of changes.

## Version Bumping Rules

| Commit Type | Version Bump | Example |
|------------|--------------|---------|
| `feat!:` or `BREAKING CHANGE:` | Major | 1.2.3 → 2.0.0 |
| `feat:` | Minor | 1.2.3 → 1.3.0 |
| `fix:` or `perf:` | Patch | 1.2.3 → 1.2.4 |
| `docs:`, `style:`, `refactor:`, `test:`, `build:`, `ci:`, `chore:` | No bump | 1.2.3 → 1.2.3 |

## Release Process

1. **Develop** your feature/fix on a feature branch
2. **Write** conventional commit messages
3. **Create** a PR to main/master
4. **Merge** the PR to main/master
5. **Automated** release pipeline:
   - Analyzes commits since last release
   - Determines version bump (major/minor/patch)
   - Runs all tests (unit + integration)
   - Creates NuGet package
   - Publishes to NuGet.org
   - Creates GitHub release with changelog
   - Updates CHANGELOG.md

## Changelog Generation

The automated release pipeline generates a categorized changelog:

```markdown
## Changes in v1.3.0

### ⚠️ BREAKING CHANGES

- feat(api)!: redesign PredictionModelBuilder interface

### ✨ Features

- feat: add GELU activation function
- feat(regression): add support for quantile regression

### 🐛 Bug Fixes

- fix(matrix): resolve determinant calculation for singular matrices
- fix(neural-networks): prevent gradient explosion

### ⚡ Performance Improvements

- perf(vector): optimize dot product calculation

### 📚 Documentation

- docs: update README with new examples
```

## Tools and Validation

### Git Hooks (Optional)

Install commitlint to validate commit messages locally:

```bash
npm install -g @commitlint/cli @commitlint/config-conventional
```

### IDE Extensions

- **VS Code**: [Conventional Commits](https://marketplace.visualstudio.com/items?itemName=vivaxy.vscode-conventional-commits)
- **JetBrains**: Use commit message templates

### Validation in CI

The commitlint workflow automatically validates all commit messages in PRs.

## Tips

1. **Make atomic commits**: One logical change per commit
2. **Write clear subjects**: Someone should understand the change without reading the body
3. **Use the body**: Explain complex changes
4. **Reference issues**: Always link to related issues
5. **Test locally**: Make sure your changes work before committing

## Questions?

- Read the [Conventional Commits specification](https://www.conventionalcommits.org/)
- Check existing commits for examples: `git log --oneline`
- Ask the team in discussions or issues

## Conventional Commits Cheat Sheet

Quick reference for commit messages:

```bash
# New feature (minor bump)
git commit -m "feat: add new activation function"

# Bug fix (patch bump)
git commit -m "fix: resolve memory leak in training loop"

# Breaking change (major bump)
git commit -m "feat!: redesign public API"
# or
git commit -m "feat: redesign API

BREAKING CHANGE: Old methods removed"

# Performance improvement (patch bump)
git commit -m "perf: optimize matrix multiplication"

# Documentation only (no bump)
git commit -m "docs: update README examples"

# With scope
git commit -m "feat(regression): add support for Ridge regression"

# Multiple lines
git commit -m "fix: resolve NaN in loss calculation

The loss function was producing NaN values when predictions
were exactly zero. Added epsilon for numerical stability.

Fixes #123"
```
