namespace AiDotNet.Serving.Persistence.Entities;

/// <summary>
/// Advisory machine activation record for soft device-binding and seat counting.
/// </summary>
public sealed class LicenseActivationEntity
{
    public Guid Id { get; set; }

    public Guid LicenseKeyId { get; set; }

    public string MachineId { get; set; } = string.Empty;

    public string? MachineName { get; set; }

    public string? Environment { get; set; }

    public DateTimeOffset FirstSeenAt { get; set; } = DateTimeOffset.UtcNow;

    public DateTimeOffset LastSeenAt { get; set; } = DateTimeOffset.UtcNow;

    public bool IsActive { get; set; } = true;
}
