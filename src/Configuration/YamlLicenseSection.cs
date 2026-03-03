namespace AiDotNet.Configuration;

/// <summary>
/// YAML configuration section for license key settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Add a <c>license</c> section to your YAML config file to set up
/// your license key and server URL. You can still override these from code.</para>
///
/// <para><b>Example YAML:</b></para>
/// <code>
/// license:
///   key: "aidn.abc123.secretXYZ"
///   serverUrl: "https://license.example.com"
///   environment: "production"
///   offlineGracePeriodDays: 14
///   enableTelemetry: true
/// </code>
/// </remarks>
public class YamlLicenseSection
{
    /// <summary>
    /// The license key string (e.g., "aidn.{id}.{secret}").
    /// </summary>
    public string Key { get; set; } = string.Empty;

    /// <summary>
    /// The license validation server URL. Null or empty for offline-only mode.
    /// </summary>
    public string? ServerUrl { get; set; }

    /// <summary>
    /// The environment label (e.g., "production", "staging", "development").
    /// </summary>
    public string? Environment { get; set; }

    /// <summary>
    /// Number of days the cached validation result remains trusted when the server is unreachable.
    /// Defaults to 7.
    /// </summary>
    public int OfflineGracePeriodDays { get; set; } = 7;

    /// <summary>
    /// Whether to send advisory machine-ID telemetry during validation. Defaults to true.
    /// </summary>
    public bool EnableTelemetry { get; set; } = true;
}
