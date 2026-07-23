using System.ComponentModel.DataAnnotations;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Request to create a Stripe Checkout session for a new subscription.
/// </summary>
public sealed class CheckoutRequest : IValidatableObject
{
    [Required(AllowEmptyStrings = false)]
    [EmailAddress]
    [StringLength(320)]
    public string Email { get; set; } = string.Empty;

    [Required(AllowEmptyStrings = false)]
    [StringLength(200, MinimumLength = 1)]
    public string CustomerName { get; set; } = string.Empty;

    [EnumDataType(typeof(SubscriptionTier))]
    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Pro;

    [Range(1, 10_000)]
    public int Seats { get; set; } = 1;

    /// <summary>
    /// Billing interval: "month" or "year". Defaults to monthly.
    /// </summary>
    [StringLength(10)]
    public string? BillingInterval { get; set; }

    public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
    {
        if (BillingInterval is not null &&
            !string.Equals(BillingInterval, "month", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(BillingInterval, "year", StringComparison.OrdinalIgnoreCase))
        {
            yield return new ValidationResult(
                "BillingInterval must be 'month' or 'year'.",
                new[] { nameof(BillingInterval) });
        }
    }
}
