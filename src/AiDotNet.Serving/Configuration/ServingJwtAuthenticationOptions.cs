namespace AiDotNet.Serving.Configuration;

/// <summary>
/// JWT bearer authentication configuration for AiDotNet.Serving.
/// </summary>
public sealed class ServingJwtAuthenticationOptions
{
    public bool Enabled { get; set; }

    /// <summary>
    /// Optional OpenID Connect authority (recommended for production IdPs).
    /// </summary>
    public string? Authority { get; set; }

    /// <summary>
    /// Optional audience to validate when using Authority or symmetric signing keys.
    /// </summary>
    public string? Audience { get; set; }

    /// <summary>
    /// Optional issuer to validate when using symmetric signing keys.
    /// </summary>
    public string? Issuer { get; set; }

    /// <summary>
    /// Optional symmetric signing key (base64 or UTF-8) for self-hosted/dev scenarios.
    /// Use Authority in production when possible.
    /// </summary>
    public string? SigningKey { get; set; }

    public bool RequireHttpsMetadata { get; set; } = true;
}

