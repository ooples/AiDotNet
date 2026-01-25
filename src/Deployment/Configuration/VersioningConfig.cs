namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model versioning - managing multiple versions of the same model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> As you improve your AI model over time, you'll have multiple versions.
/// Versioning helps you manage these versions, allowing you to:
/// - Keep track of which version is deployed
/// - Roll back to a previous version if needed
/// - Gradually transition users from old to new versions
/// - Compare performance between versions
///
/// Version Format: Follows semantic versioning (e.g., "1.2.3")
/// - Major version: Breaking changes (1.0.0 → 2.0.0)
/// - Minor version: New features, backwards compatible (1.0.0 → 1.1.0)
/// - Patch version: Bug fixes (1.0.0 → 1.0.1)
///
/// You can also use "latest" to always get the newest version, or "stable" for the
/// most reliable production version.
/// </para>
/// </remarks>
public class VersioningConfig
{
    /// <summary>
    /// Gets or sets whether versioning is enabled (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true to enable version management, false to use a single model version.
    /// Versioning is recommended for production systems.
    /// </para>
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the default version to use when none is specified (default: "latest").
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When no version is specified, this version will be used.
    /// "latest" uses the newest version, or specify a version like "1.2.3" for a specific one.
    /// </para>
    /// </remarks>
    public string DefaultVersion { get; set; } = "latest";

    /// <summary>
    /// Gets or sets whether to allow automatic version upgrades (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, requests for "1.x" automatically use the latest 1.x version.
    /// False means you must specify exact versions. Recommended to keep false for predictability.
    /// </para>
    /// </remarks>
    public bool AllowAutoUpgrade { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of versions to keep in history (default: 5).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many old versions to keep. Older versions are deleted to save disk space.
    /// 5 is usually enough for rollback purposes.
    /// </para>
    /// </remarks>
    public int MaxVersionHistory { get; set; } = 3;

    /// <summary>
    /// Alias for MaxVersionHistory for more intuitive access.
    /// </summary>
    public int MaxVersionsPerModel
    {
        get => MaxVersionHistory;
        set => MaxVersionHistory = value;
    }

    /// <summary>
    /// Gets or sets whether to automatically clean up old versions when MaxVersionHistory is exceeded (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, old model versions are automatically deleted when
    /// you exceed MaxVersionHistory. When false, you must manually delete old versions.
    /// Recommended to keep true for automatic disk space management.
    /// </para>
    /// </remarks>
    public bool AutoCleanup { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track which version is used for each inference (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records which model version was used for each prediction.
    /// Useful for debugging and analytics. Small performance overhead.
    /// </para>
    /// </remarks>
    public bool TrackVersionUsage { get; set; } = true;

    /// <summary>
    /// Gets or sets the version metadata dictionary for storing additional version information.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Store custom information about each version (release date, description, etc.).
    /// Useful for documentation and tracking.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> VersionMetadata { get; set; } = new();
}
