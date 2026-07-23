namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Administrative view of a license key, including activation details.
/// </summary>
public sealed class LicenseInfo
{
    public Guid Id { get; set; }

    public string KeyId { get; set; } = string.Empty;

    public string CustomerName { get; set; } = string.Empty;

    public string? CustomerEmail { get; set; }

    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    public int MaxSeats { get; set; } = 1;

    public int SeatsUsed { get; set; }

    public DateTimeOffset CreatedAt { get; set; }

    public DateTimeOffset? ExpiresAt { get; set; }

    public DateTimeOffset? RevokedAt { get; set; }

    public string? Environment { get; set; }

    public string? Notes { get; set; }

    public List<LicenseActivationInfo> Activations { get; set; } = new();
}

/// <summary>
/// Advisory machine activation record for display in admin views.
/// </summary>
public sealed class LicenseActivationInfo
{
    public string MachineId { get; set; } = string.Empty;

    public string? MachineName { get; set; }

    public string? Environment { get; set; }

    public DateTimeOffset FirstSeenAt { get; set; }

    public DateTimeOffset LastSeenAt { get; set; }

    public bool IsActive { get; set; }
}
