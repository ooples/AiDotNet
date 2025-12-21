using AiDotNet.Serving.Security;
using System.ComponentModel.DataAnnotations;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Request to create an API key.
/// </summary>
public sealed class ApiKeyCreateRequest : IValidatableObject
{
    [Required(AllowEmptyStrings = false)]
    [StringLength(200, MinimumLength = 1)]
    public string Name { get; set; } = string.Empty;

    [EnumDataType(typeof(SubscriptionTier))]
    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    public ApiKeyScopes? Scopes { get; set; } = null;

    public DateTimeOffset? ExpiresAt { get; set; } = null;

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
