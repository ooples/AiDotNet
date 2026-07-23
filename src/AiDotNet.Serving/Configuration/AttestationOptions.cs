namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for client attestation in premium deployment tiers.
/// </summary>
public class AttestationOptions
{
    /// <summary>
    /// Gets or sets whether to allow unverified attestation when running in the Development environment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is convenient for local testing, but should be disabled in production.
    /// </remarks>
    public bool AllowUnverifiedAttestationInDevelopment { get; set; } = true;

    /// <summary>
    /// Gets or sets an optional static test token.
    /// </summary>
    /// <remarks>
    /// This is intended for test environments. Production deployments should prefer <see cref="Jwt"/>.
    /// </remarks>
    public string? StaticTestToken { get; set; } = null;

    /// <summary>
    /// Gets or sets JWT verification options for signed attestation tokens.
    /// </summary>
    public JwtAttestationOptions? Jwt { get; set; } = null;

    /// <summary>
    /// Gets or sets the allowed platforms (case-insensitive). Empty means allow any platform.
    /// </summary>
    public string[] AllowedPlatforms { get; set; } = [];

    /// <summary>
    /// Gets or sets the allowed TEE types (case-insensitive). Empty means allow any TEE type.
    /// </summary>
    public string[] AllowedTeeTypes { get; set; } = [];
}
