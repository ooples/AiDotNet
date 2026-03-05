using AiDotNet.Serving.Security.Licensing;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Public endpoints for Stripe Checkout, Customer Portal, and webhook handling.
/// Stripe webhook signature verification provides authentication for webhook events.
/// </summary>
[ApiController]
[Route("api")]
public sealed class StripeController : ControllerBase
{
    private readonly IStripeService _stripeService;
    private readonly ILogger<StripeController> _logger;

    public StripeController(IStripeService stripeService, ILogger<StripeController> logger)
    {
        Guard.NotNull(stripeService);
        Guard.NotNull(logger);
        _stripeService = stripeService;
        _logger = logger;
    }

    /// <summary>
    /// Creates a Stripe Checkout session and returns the URL to redirect the customer to.
    /// </summary>
    [HttpPost("checkout/session")]
    [ProducesResponseType(typeof(CheckoutResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> CreateCheckoutSession(
        [FromBody] CheckoutRequest request,
        CancellationToken cancellationToken)
    {
        try
        {
            string url = await _stripeService.CreateCheckoutSessionAsync(request, cancellationToken)
                .ConfigureAwait(false);
            return Ok(new CheckoutResponse { CheckoutUrl = url });
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "Invalid checkout request");
            return BadRequest(new { error = ex.Message });
        }
    }

    /// <summary>
    /// Creates a Stripe Customer Portal session for managing subscriptions.
    /// </summary>
    [Authorize]
    [HttpPost("checkout/portal")]
    [ProducesResponseType(typeof(CheckoutResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status401Unauthorized)]
    public async Task<IActionResult> CreatePortalSession(
        [FromBody] PortalRequest request,
        CancellationToken cancellationToken)
    {
        if (request is null || string.IsNullOrWhiteSpace(request.StripeCustomerId))
        {
            return BadRequest(new { error = "StripeCustomerId is required." });
        }

        // Validate the customer ID belongs to the authenticated user.
        // The caller's identity (from JWT sub claim) must match the stored customer mapping.
        string? userId = User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value
            ?? User.FindFirst("sub")?.Value;

        if (string.IsNullOrEmpty(userId))
        {
            return Unauthorized(new { error = "Cannot determine authenticated user identity." });
        }

        try
        {
            // Verify ownership: the Stripe customer ID must be associated with this user
            bool isOwner = await _stripeService.ValidateCustomerOwnershipAsync(
                userId, request.StripeCustomerId, cancellationToken).ConfigureAwait(false);

            if (!isOwner)
            {
                _logger.LogWarning("User {UserId} attempted portal access for unowned customer {CustomerId}",
                    userId, request.StripeCustomerId);
                return StatusCode(StatusCodes.Status403Forbidden,
                    new { error = "You do not have access to this billing portal." });
            }

            string url = await _stripeService.CreatePortalSessionAsync(request.StripeCustomerId, cancellationToken)
                .ConfigureAwait(false);
            return Ok(new CheckoutResponse { CheckoutUrl = url });
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "Invalid portal request");
            return BadRequest(new { error = ex.Message });
        }
    }

    /// <summary>
    /// Stripe webhook endpoint. Validates the webhook signature and processes events.
    /// </summary>
    [HttpPost("webhooks/stripe")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> HandleWebhook(CancellationToken cancellationToken)
    {
        string json;
        using (var reader = new StreamReader(HttpContext.Request.Body))
        {
            json = await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
        }

        string signature = HttpContext.Request.Headers["Stripe-Signature"].ToString();

        if (string.IsNullOrWhiteSpace(signature))
        {
            return BadRequest(new { error = "Missing Stripe-Signature header." });
        }

        try
        {
            await _stripeService.HandleWebhookAsync(json, signature, cancellationToken)
                .ConfigureAwait(false);
            return Ok();
        }
        catch (Stripe.StripeException ex)
        {
            _logger.LogWarning(ex, "Stripe webhook signature verification failed");
            return BadRequest(new { error = "Invalid webhook signature." });
        }
    }
}

/// <summary>
/// Request to create a Stripe Customer Portal session.
/// </summary>
public sealed class PortalRequest
{
    public string StripeCustomerId { get; set; } = string.Empty;
}
