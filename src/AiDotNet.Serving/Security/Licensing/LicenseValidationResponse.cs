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
}
