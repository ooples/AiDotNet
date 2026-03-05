namespace AiDotNet.Serving.Persistence.Entities;

/// <summary>
/// Persisted Stripe customer record for linking Stripe customers to licenses.
/// </summary>
public sealed class StripeCustomerEntity
{
    public Guid Id { get; set; }

    public string StripeCustomerId { get; set; } = string.Empty;

    public string Email { get; set; } = string.Empty;

    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// The application user ID (from JWT sub claim) that owns this Stripe customer record.
    /// Set during checkout when ClientReferenceId is provided.
    /// </summary>
    public string? UserId { get; set; }

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}
