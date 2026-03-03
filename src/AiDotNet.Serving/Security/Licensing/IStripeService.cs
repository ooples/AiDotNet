namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Service interface for Stripe payment integration including checkout, portal, and webhook handling.
/// </summary>
public interface IStripeService
{
    /// <summary>
    /// Creates a Stripe Checkout session for a new subscription.
    /// </summary>
    /// <returns>The Stripe Checkout URL to redirect the customer to.</returns>
    Task<string> CreateCheckoutSessionAsync(CheckoutRequest request, CancellationToken ct = default);

    /// <summary>
    /// Creates a Stripe Customer Portal session for managing an existing subscription.
    /// </summary>
    /// <returns>The Customer Portal URL to redirect the customer to.</returns>
    Task<string> CreatePortalSessionAsync(string stripeCustomerId, CancellationToken ct = default);

    /// <summary>
    /// Processes incoming Stripe webhook events (checkout completed, subscription updated/deleted, invoice failures).
    /// </summary>
    Task HandleWebhookAsync(string json, string signature, CancellationToken ct = default);
}
