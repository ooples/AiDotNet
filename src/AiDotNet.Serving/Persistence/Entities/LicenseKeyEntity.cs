using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Persistence.Entities;

/// <summary>
/// Persisted license key record for subscription and seat enforcement.
/// </summary>
public sealed class LicenseKeyEntity
{
    public Guid Id { get; set; }

    public string KeyId { get; set; } = string.Empty;

    public byte[] Salt { get; set; } = Array.Empty<byte>();

    public byte[] Hash { get; set; } = Array.Empty<byte>();

    public int Pbkdf2Iterations { get; set; } = 210_000;

    public string CustomerName { get; set; } = string.Empty;

    public string? CustomerEmail { get; set; }

    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    public int MaxSeats { get; set; } = 1;

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;

    public DateTimeOffset? ExpiresAt { get; set; }

    public DateTimeOffset? RevokedAt { get; set; }

    public string? Environment { get; set; }

    public string? Notes { get; set; }

    /// <summary>
    /// Server-side escrow secret (32 bytes) used to compute decryption tokens.
    /// Generated at license creation time. Never sent to the client directly.
    /// </summary>
    public byte[] EscrowSecret { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Validates crypto field invariants before persistence.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when crypto fields are invalid.</exception>
    public void ValidateCryptoFields()
    {
        if (Pbkdf2Iterations < 100_000)
        {
            throw new InvalidOperationException(
                $"Pbkdf2Iterations must be at least 100,000, got {Pbkdf2Iterations}.");
        }

        if (EscrowSecret.Length > 0 && EscrowSecret.Length != 32)
        {
            throw new InvalidOperationException(
                $"EscrowSecret must be exactly 32 bytes when set, got {EscrowSecret.Length}.");
        }

        if (Salt.Length > 0 && Salt.Length < 16)
        {
            throw new InvalidOperationException(
                $"Salt must be at least 16 bytes, got {Salt.Length}.");
        }
    }
}
