namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for API key based tier access in AiDotNet.Serving.
/// </summary>
public sealed class ServingApiKeyOptions
{
    /// <summary>
    /// Header name used to pass API keys.
    /// </summary>
    public string HeaderName { get; set; } = "X-AiDotNet-Api-Key";

    /// <summary>
    /// PBKDF2 iteration count used when hashing API keys.
    /// </summary>
    /// <remarks>
    /// The default is intentionally conservative for production security. Override for performance only with care.
    /// </remarks>
    public int Pbkdf2Iterations { get; set; } = 600_000;

    /// <summary>
    /// One or more active HMAC secrets used to derive lookup identifiers from API keys.
    /// New secrets should be prepended to rotate while keeping older secrets to validate existing keys.
    /// </summary>
    /// <remarks>
    /// Store secrets in a secure provider (environment variables, secret manager, key vault) and avoid committing them
    /// to source control in plaintext.
    /// </remarks>
    public List<string> HmacSecrets { get; set; } = new();

    /// <summary>
    /// Optional static (config-backed) API keys for self-hosted/dev scenarios.
    /// These should be stored as hashes/identifiers (never plaintext).
    /// </summary>
    public List<StaticApiKeyOptions> StaticApiKeys { get; set; } = new();
}
