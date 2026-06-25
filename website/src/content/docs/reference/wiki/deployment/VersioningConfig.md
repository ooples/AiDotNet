---
title: "VersioningConfig"
description: "Configuration for model versioning - managing multiple versions of the same model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for model versioning - managing multiple versions of the same model.

## For Beginners

As you improve your AI model over time, you'll have multiple versions.
Versioning helps you manage these versions, allowing you to:

- Keep track of which version is deployed
- Roll back to a previous version if needed
- Gradually transition users from old to new versions
- Compare performance between versions

Version Format: Follows semantic versioning (e.g., "1.2.3")

- Major version: Breaking changes (1.0.0 → 2.0.0)
- Minor version: New features, backwards compatible (1.0.0 → 1.1.0)
- Patch version: Bug fixes (1.0.0 → 1.0.1)

You can also use "latest" to always get the newest version, or "stable" for the
most reliable production version.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowAutoUpgrade` | Gets or sets whether to allow automatic version upgrades (default: false). |
| `AutoCleanup` | Gets or sets whether to automatically clean up old versions when MaxVersionHistory is exceeded (default: true). |
| `DefaultVersion` | Gets or sets the default version to use when none is specified (default: "latest"). |
| `Enabled` | Gets or sets whether versioning is enabled (default: true). |
| `MaxVersionHistory` | Gets or sets the maximum number of versions to keep in history (default: 5). |
| `MaxVersionsPerModel` | Alias for MaxVersionHistory for more intuitive access. |
| `TrackVersionUsage` | Gets or sets whether to track which version is used for each inference (default: true). |
| `VersionMetadata` | Gets or sets the version metadata dictionary for storing additional version information. |

