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
}
