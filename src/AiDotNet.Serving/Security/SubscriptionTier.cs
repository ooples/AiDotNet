namespace AiDotNet.Serving.Security;

/// <summary>
/// Subscription tiers used for policy enforcement.
/// </summary>
public enum SubscriptionTier
{
    /// <summary>
    /// Free/open-source tier (Option A): server-side inference only.
    /// </summary>
    Free,

    /// <summary>
    /// Pro tier (Option B): allows model artifact download.
    /// </summary>
    Pro,

    /// <summary>
    /// Enterprise tier (Option C): encrypted artifact + attested key release.
    /// </summary>
    Enterprise
}

