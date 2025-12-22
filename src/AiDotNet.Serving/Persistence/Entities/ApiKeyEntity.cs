using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Persistence.Entities;

/// <summary>
/// Persisted API key record for tier enforcement.
/// </summary>
public sealed class ApiKeyEntity
{
    public Guid Id { get; set; }

    public string KeyId { get; set; } = string.Empty;

    public string Name { get; set; } = string.Empty;

    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    public ApiKeyScopes Scopes { get; set; } = ApiKeyScopes.None;

    public byte[] Salt { get; set; } = Array.Empty<byte>();

    public byte[] Hash { get; set; } = Array.Empty<byte>();

    public int Pbkdf2Iterations { get; set; } = 210_000;

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;

    public DateTimeOffset? ExpiresAt { get; set; }

    public DateTimeOffset? RevokedAt { get; set; }
}

