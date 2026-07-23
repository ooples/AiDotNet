namespace AiDotNet.Serving.Security;

/// <summary>
/// Provides policies for subscription tiers.
/// </summary>
public interface ITierPolicyProvider
{
    /// <summary>
    /// Gets the policy for the specified tier.
    /// </summary>
    TierPolicy GetPolicy(SubscriptionTier tier);
}

