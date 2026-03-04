using System.Text.Json.Serialization;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// The possible statuses of a license validation result.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum LicenseKeyStatus
{
    /// <summary>License is valid and active.</summary>
    Active,

    /// <summary>License key is invalid or not found.</summary>
    Invalid,

    /// <summary>License has been revoked.</summary>
    Revoked,

    /// <summary>License has expired.</summary>
    Expired,

    /// <summary>All available seats are already in use.</summary>
    SeatLimitReached
}

/// <summary>
/// Response returned by the license validation endpoint.
/// </summary>
public sealed class LicenseValidationResponse
{
    public LicenseKeyStatus Status { get; set; }

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
