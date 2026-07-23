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
    /// Gets a copy of the server-side decryption token for Layer 2 key escrow, or null if not available.
    /// This token is cached alongside the validation result for offline use.
    /// Returns a defensive copy to prevent external mutation of the cached token.
    /// </summary>
    public byte[]? DecryptionToken => _decryptionToken is not null ? (byte[])_decryptionToken.Clone() : null;

    private readonly byte[]? _decryptionToken;

    /// <summary>
    /// Gets the namespace-prefixed capability strings (e.g., <c>tensors:save</c>,
    /// <c>tensors:load</c>, <c>model:save</c>, <c>model:load</c>) the server
    /// returned for the validated license. Capability gating is performed
    /// client-side by exact ordinal lookup. May be empty when validating
    /// against an older server that does not yet attach the
    /// <c>capabilities</c> field — issue #1195's coordinated rollout
    /// guarantees the server is updated before any client requires a
    /// specific capability.
    /// </summary>
    public IReadOnlyList<string> Capabilities => _capabilities;

    private readonly IReadOnlyList<string> _capabilities;

    /// <summary>
    /// Returns true if the server granted the named capability for this license.
    /// Capability strings are matched ordinally (case-sensitive). Returns false
    /// when the capability list is empty (older server) or the name is not present.
    /// </summary>
    /// <param name="capability">The namespace-prefixed capability to check (e.g., <c>tensors:save</c>).</param>
    public bool HasCapability(string capability)
    {
        if (string.IsNullOrEmpty(capability)) return false;
        for (int i = 0; i < _capabilities.Count; i++)
        {
            if (string.Equals(_capabilities[i], capability, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

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
        byte[]? decryptionToken = null,
        IReadOnlyList<string>? capabilities = null)
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
        _decryptionToken = decryptionToken is not null ? (byte[])decryptionToken.Clone() : null;
        // Defensive copy: if a caller mutates the array they passed in,
        // we don't want the cached LicenseValidationResult to silently
        // re-grant or revoke capabilities behind the guard's back.
        _capabilities = capabilities is null
            ? System.Array.Empty<string>()
            : new List<string>(capabilities).AsReadOnly();
    }
}
