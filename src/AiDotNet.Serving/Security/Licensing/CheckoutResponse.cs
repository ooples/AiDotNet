namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Response containing the Stripe Checkout URL for the customer to complete payment.
/// </summary>
public sealed class CheckoutResponse
{
    public string CheckoutUrl { get; set; } = string.Empty;
}
