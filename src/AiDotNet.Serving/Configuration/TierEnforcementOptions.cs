using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for monetization tier enforcement.
/// </summary>
public class TierEnforcementOptions
{
    /// <summary>
    /// Gets or sets whether tier enforcement is enabled.
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the HTTP header name used to specify the request tier (development override only).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> In production, tiers should come from authenticated identity (API keys or an identity provider),
    /// not from a user-controlled header. This header is only intended for local development and testing scenarios.
    /// </remarks>
    public string TierHeaderName { get; set; } = "X-AiDotNet-Tier";

    /// <summary>
    /// Gets or sets the default tier used when the header is missing.
    /// </summary>
    public SubscriptionTier DefaultTier { get; set; } = SubscriptionTier.Free;
}

