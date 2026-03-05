namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for Stripe payment integration.
/// </summary>
public class StripeOptions
{
    /// <summary>
    /// Gets or sets the Stripe secret API key.
    /// </summary>
    public string SecretKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the Stripe webhook signing secret for signature verification.
    /// </summary>
    public string WebhookSigningSecret { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the Stripe Price ID for Pro tier monthly subscription.
    /// </summary>
    public string ProPriceId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the Stripe Price ID for Pro tier annual subscription.
    /// </summary>
    public string ProAnnualPriceId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the Stripe Price ID for Enterprise tier subscription.
    /// </summary>
    public string EnterprisePriceId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the URL to redirect to after a successful checkout.
    /// </summary>
    public string SuccessUrl { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the URL to redirect to after a cancelled checkout.
    /// </summary>
    public string CancelUrl { get; set; } = string.Empty;
}
