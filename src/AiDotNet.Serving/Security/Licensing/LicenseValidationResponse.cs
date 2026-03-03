namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Response returned by the license validation endpoint.
/// </summary>
public sealed class LicenseValidationResponse
{
    public string Status { get; set; } = string.Empty;

    public string? Tier { get; set; }

    public DateTimeOffset? ExpiresAt { get; set; }

    public int SeatsUsed { get; set; }

    public int? SeatsMax { get; set; }

    public string? Message { get; set; }

    /// <summary>
    /// Base64-encoded decryption token derived from the server-side escrow secret.
    /// Only returned when the license status is Active. Used for Layer 2 key escrow.
    /// </summary>
    public string? DecryptionToken { get; set; }
}
