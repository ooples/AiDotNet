using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Represents a hashed API key entry stored in configuration for self-hosted/dev scenarios.
/// </summary>
public sealed class StaticApiKeyOptions
{
    public string Prefix { get; set; } = string.Empty;

    public string HmacIdBase64 { get; set; } = string.Empty;

    public string Pbkdf2SaltBase64 { get; set; } = string.Empty;

    public string Pbkdf2HashBase64 { get; set; } = string.Empty;

    public int Pbkdf2Iterations { get; set; } = 600_000;

    public ServingTier Tier { get; set; } = ServingTier.Free;

    public DateTimeOffset? ExpiresAt { get; set; }

    public bool Revoked { get; set; }
}
