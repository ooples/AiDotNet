using System.ComponentModel.DataAnnotations;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Request to create a new license key.
/// </summary>
public sealed class LicenseCreateRequest : IValidatableObject
{
    [Required(AllowEmptyStrings = false)]
    [StringLength(200, MinimumLength = 1)]
    [RegularExpression(@".*\S.*", ErrorMessage = "CustomerName cannot be whitespace only.")]
    public string CustomerName { get; set; } = string.Empty;

    [EmailAddress]
    [StringLength(320)]
    public string? CustomerEmail { get; set; }

    [EnumDataType(typeof(SubscriptionTier))]
    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    [Range(1, 10_000)]
    public int MaxSeats { get; set; } = 1;

    public DateTimeOffset? ExpiresAt { get; set; }

    [StringLength(64)]
    public string? Environment { get; set; }

    [StringLength(2000)]
    public string? Notes { get; set; }

    public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
    {
        if (ExpiresAt is not null && ExpiresAt <= DateTimeOffset.UtcNow)
        {
            yield return new ValidationResult(
                "ExpiresAt must be a future time.",
                new[] { nameof(ExpiresAt) });
        }
    }
}
