namespace AiDotNet.Serving.Enums;

/// <summary>
/// Represents the status of a Stripe subscription.
/// </summary>
public enum StripeSubscriptionStatus
{
    Active,
    PastDue,
    Cancelled,
    Incomplete,
    Trialing,
    Paused
}
