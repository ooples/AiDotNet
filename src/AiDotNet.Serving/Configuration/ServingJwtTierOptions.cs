namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for mapping JWT claims to tier context.
/// </summary>
public sealed class ServingJwtTierOptions
{
    /// <summary>
    /// Claim type used to represent the subscription tier (e.g., "tier").
    /// </summary>
    public string TierClaimType { get; set; } = "tier";
}

