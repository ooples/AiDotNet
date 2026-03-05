using AiDotNet.Serving.Enums;

namespace AiDotNet.Serving.Persistence.Entities;

/// <summary>
/// Persisted Stripe subscription record linking a subscription to a customer and license key.
/// </summary>
public sealed class StripeSubscriptionEntity
{
    public Guid Id { get; set; }

    public string StripeSubscriptionId { get; set; } = string.Empty;

    public string StripeCustomerId { get; set; } = string.Empty;

    public Guid? LicenseKeyId { get; set; }

    public string StripePriceId { get; set; } = string.Empty;

    public StripeSubscriptionStatus Status { get; set; } = StripeSubscriptionStatus.Active;

    public DateTimeOffset CurrentPeriodStart { get; set; }

    public DateTimeOffset CurrentPeriodEnd { get; set; }

    public DateTimeOffset? CancelledAt { get; set; }

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}
