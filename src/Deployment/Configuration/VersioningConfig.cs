namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model versioning - managing multiple versions of the same model.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> As you improve your AI model over time, you'll have multiple versions.
/// Versioning helps you manage these versions, allowing you to:
/// - Keep track of which version is deployed
/// - Roll back to a previous version if needed
/// - Gradually transition users from old to new versions
/// - Compare performance between versions
///
/// **Version Format:** Follows semantic versioning (e.g., "1.2.3")
/// - Major version: Breaking changes (1.0.0 → 2.0.0)
/// - Minor version: New features, backwards compatible (1.0.0 → 1.1.0)
/// - Patch version: Bug fixes (1.0.0 → 1.0.1)
///
/// You can also use "latest" to always get the newest version, or "stable" for the
/// most reliable production version.
/// </remarks>
public class VersioningConfig
{
    /// <summary>
    /// Gets or sets whether versioning is enabled (default: true).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the default version to use when none is specified (default: "latest").
    /// Can be a specific version like "1.2.3" or special values like "latest" or "stable".
    /// </summary>
    public string DefaultVersion { get; set; } = "latest";

    /// <summary>
    /// Gets or sets whether to allow automatic version upgrades (default: false).
    /// If true, requests for "1.x" will use the latest 1.x version automatically.
    /// </summary>
    public bool AllowAutoUpgrade { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of versions to keep in history (default: 5).
    /// Older versions are removed to save disk space.
    /// </summary>
    public int MaxVersionHistory { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to track which version is used for each inference (default: true).
    /// Useful for debugging and analytics.
    /// </summary>
    public bool TrackVersionUsage { get; set; } = true;

    /// <summary>
    /// Gets or sets the version metadata dictionary for storing additional version information.
    /// </summary>
    public Dictionary<string, string> VersionMetadata { get; set; } = new();

    /// <summary>
    /// Creates a disabled versioning configuration (single model, no versioning).
    /// </summary>
    public static VersioningConfig Disabled()
    {
        return new VersioningConfig
        {
            Enabled = false,
            DefaultVersion = "1.0.0"
        };
    }

    /// <summary>
    /// Creates a strict versioning configuration (no auto-upgrades, explicit versions only).
    /// </summary>
    public static VersioningConfig Strict()
    {
        return new VersioningConfig
        {
            Enabled = true,
            AllowAutoUpgrade = false,
            TrackVersionUsage = true
        };
    }

    /// <summary>
    /// Creates a flexible versioning configuration (auto-upgrades enabled, uses latest by default).
    /// </summary>
    public static VersioningConfig Flexible()
    {
        return new VersioningConfig
        {
            Enabled = true,
            DefaultVersion = "latest",
            AllowAutoUpgrade = true
        };
    }
}
