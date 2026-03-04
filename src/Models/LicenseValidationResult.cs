using AiDotNet.Enums;

namespace AiDotNet.Models;

/// <summary>
/// Contains the result of a license key validation attempt.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After the license validator contacts the server (or checks its cache),
/// this object tells you whether the key is valid, what tier the user is on, how many seats are used,
/// and when the key expires.</para>
/// </remarks>
public sealed class LicenseValidationResult
{
    /// <summary>
    /// Gets the validation status of the license key.
    /// </summary>
    public LicenseKeyStatus Status { get; }

    /// <summary>
    /// Gets the subscription tier associated with this license, or null if unknown.
    /// </summary>
    public string? Tier { get; }

    /// <summary>
    /// Gets the expiration date of the license, or null if it does not expire.
    /// </summary>
    public DateTimeOffset? ExpiresAt { get; }

    /// <summary>
    /// Gets the number of seats currently in use for this license.
    /// </summary>
    public int SeatsUsed { get; }

    /// <summary>
    /// Gets the maximum number of seats allowed for this license, or null if unlimited.
    /// </summary>
    public int? SeatsMax { get; }

    /// <summary>
    /// Gets the UTC timestamp of when this validation was performed.
    /// </summary>
    public DateTimeOffset ValidatedAt { get; }

    /// <summary>
    /// Gets an optional human-readable message from the server.
    /// </summary>
    public string? Message { get; }

    /// <summary>
    /// Gets the server-side decryption token for Layer 2 key escrow, or null if not available.
    /// This token is cached alongside the validation result for offline use.
    /// </summary>
    public byte[]? DecryptionToken { get; }

    /// <summary>
    /// Creates a new <see cref="LicenseValidationResult"/>.
    /// </summary>
    public LicenseValidationResult(
        LicenseKeyStatus status,
        string? tier = null,
        DateTimeOffset? expiresAt = null,
        int seatsUsed = 0,
        int? seatsMax = null,
        DateTimeOffset? validatedAt = null,
        string? message = null,
        byte[]? decryptionToken = null)
    {
        if (seatsUsed < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(seatsUsed), "Seats used cannot be negative.");
        }

        if (seatsMax.HasValue && seatsMax.Value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(seatsMax), "Seats max cannot be negative.");
        }

        Status = status;
        Tier = tier;
        ExpiresAt = expiresAt;
        SeatsUsed = seatsUsed;
        SeatsMax = seatsMax;
        ValidatedAt = validatedAt ?? DateTimeOffset.UtcNow;
        Message = message;
        DecryptionToken = decryptionToken is not null ? (byte[])decryptionToken.Clone() : null;
    }
}
