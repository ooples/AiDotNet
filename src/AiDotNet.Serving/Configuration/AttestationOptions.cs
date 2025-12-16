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
    /// Gets or sets an optional static test token that can be used for non-development environments.
    /// </summary>
    /// <remarks>
    /// This is a placeholder for real attestation providers (e.g., device/OS attestation, enclave attestation).
    /// </remarks>
    public string? StaticTestToken { get; set; } = null;
}

