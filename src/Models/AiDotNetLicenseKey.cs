using AiDotNet.Validation;

namespace AiDotNet.Models;

/// <summary>
/// Represents a license key for AiDotNet model encryption and online validation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you purchase a license for AiDotNet, you receive a license key.
/// This class wraps that key along with optional configuration for connecting to a license server.
/// You pass it to <c>AiModelBuilder</c> so encrypted models can be loaded and saved.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// // Minimal: just the key (offline-only, no server validation)
/// var license = new AiDotNetLicenseKey("aidn.abc123.secretXYZ");
///
/// // Full: with server validation
/// var license = new AiDotNetLicenseKey("aidn.abc123.secretXYZ")
/// {
///     ServerUrl = "https://license.example.com",
///     Environment = "production",
///     OfflineGracePeriod = TimeSpan.FromDays(14),
///     EnableTelemetry = true
/// };
///
/// var builder = new AiModelBuilder&lt;double, double[], double&gt;(license);
/// </code>
/// </remarks>
public sealed class AiDotNetLicenseKey
{
    /// <summary>
    /// Gets the license key string (e.g., "aidn.{id}.{secret}").
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// Gets or sets the URL of the license validation server.
    /// When null, the license operates in offline-only mode (no server validation).
    /// </summary>
    public string? ServerUrl { get; set; }

    /// <summary>
    /// Gets or sets the environment label sent during validation (e.g., "production", "staging", "development").
    /// </summary>
    public string? Environment { get; set; }

    /// <summary>
    /// Gets or sets the duration that a cached validation result remains trusted when the server is unreachable.
    /// Defaults to 7 days.
    /// </summary>
    public TimeSpan OfflineGracePeriod { get; set; } = TimeSpan.FromDays(7);

    /// <summary>
    /// Gets or sets whether advisory machine-ID telemetry is sent to the license server during validation.
    /// Defaults to true.
    /// </summary>
    public bool EnableTelemetry { get; set; } = true;

    /// <summary>
    /// Creates a new <see cref="AiDotNetLicenseKey"/> with the specified key string.
    /// </summary>
    /// <param name="key">The license key string. Must not be null or whitespace.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="key"/> is empty or whitespace.</exception>
    public AiDotNetLicenseKey(string key)
    {
        Guard.NotNullOrWhiteSpace(key);
        Key = key;
    }
}
