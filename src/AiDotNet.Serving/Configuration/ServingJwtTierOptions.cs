namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for mapping JWT claims to tier context.
/// </summary>
public sealed class ServingJwtTierOptions
{
    /// <summary>
    /// Gets or sets the claim type that carries a JWT caller's subscription tier (for example, <c>"tier"</c>).
    /// Its value is a <see cref="AiDotNet.Serving.Security.SubscriptionTier"/> name, matched case-insensitively;
    /// a missing or unknown value gets the default tier.
    /// </summary>
    public string TierClaimType { get; set; } = "tier";

    /// <summary>
    /// Validates these settings and throws <see cref="InvalidOperationException"/> naming the first invalid one.
    /// </summary>
    /// <param name="configurationPath">The configuration path used in error messages.</param>
    public void Validate(string configurationPath)
    {
        if (string.IsNullOrWhiteSpace(TierClaimType))
        {
            throw new InvalidOperationException($"{configurationPath}:TierClaimType must be a non-empty string.");
        }
    }
}
