namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Response for license key creation. The license key string is returned only once.
/// </summary>
public sealed class LicenseCreateResponse
{
    public Guid Id { get; set; }

    public string LicenseKey { get; set; } = string.Empty;

    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    public int MaxSeats { get; set; } = 1;

    public DateTimeOffset? ExpiresAt { get; set; }

    public DateTimeOffset CreatedAt { get; set; }
}
