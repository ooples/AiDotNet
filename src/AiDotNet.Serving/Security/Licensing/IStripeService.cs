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

    /// <summary>
    /// Validates that the given Stripe customer ID is associated with the authenticated user.
    /// </summary>
    /// <param name="userId">The authenticated user's ID.</param>
    /// <param name="stripeCustomerId">The Stripe customer ID to validate.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>True if the customer ID belongs to this user.</returns>
    Task<bool> ValidateCustomerOwnershipAsync(string userId, string stripeCustomerId, CancellationToken ct = default);
}
