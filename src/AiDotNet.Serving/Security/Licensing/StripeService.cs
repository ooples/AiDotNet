using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Enums;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Persistence.Entities;
using AiDotNet.Validation;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Stripe;
using Stripe.Checkout;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Stripe payment integration service that handles checkout sessions, customer portal, and webhook events.
/// </summary>
internal sealed class StripeService : IStripeService
{
    private readonly StripeOptions _options;
    private readonly ILicenseService _licenseService;
    private readonly ServingDbContext _db;
    private readonly ILogger<StripeService> _logger;

    public StripeService(
        IOptions<StripeOptions> options,
        ILicenseService licenseService,
        ServingDbContext db,
        ILogger<StripeService> logger)
    {
        Guard.NotNull(options);
        Guard.NotNull(licenseService);
        Guard.NotNull(db);
        Guard.NotNull(logger);

        _options = options.Value;
        _licenseService = licenseService;
        _db = db;
        _logger = logger;

        StripeConfiguration.ApiKey = _options.SecretKey;
    }

    public async Task<string> CreateCheckoutSessionAsync(CheckoutRequest request, CancellationToken ct = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (string.IsNullOrWhiteSpace(request.Email))
        {
            throw new ArgumentException("Email is required.", nameof(request));
        }

        if (string.IsNullOrWhiteSpace(request.CustomerName))
        {
            throw new ArgumentException("CustomerName is required.", nameof(request));
        }

        if (request.Seats < 1 || request.Seats > 10_000)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Seats must be between 1 and 10,000.");
        }

        string priceId = ResolvePriceId(request.Tier, request.BillingInterval);

        var sessionOptions = new SessionCreateOptions
        {
            Mode = "subscription",
            CustomerEmail = request.Email,
            LineItems = new List<SessionLineItemOptions>
            {
                new SessionLineItemOptions
                {
                    Price = priceId,
                    Quantity = request.Seats
                }
            },
            SuccessUrl = _options.SuccessUrl + "?session_id={CHECKOUT_SESSION_ID}",
            CancelUrl = _options.CancelUrl,
            Metadata = new Dictionary<string, string>
            {
                ["customer_name"] = request.CustomerName,
                ["tier"] = request.Tier.ToString(),
                ["seats"] = request.Seats.ToString()
            }
        };

        var sessionService = new SessionService();
        var session = await sessionService.CreateAsync(sessionOptions, cancellationToken: ct).ConfigureAwait(false);

        _logger.LogInformation(
            "Created Stripe Checkout session {SessionId} (Tier={Tier}, Seats={Seats})",
            session.Id, request.Tier, request.Seats);

        return session.Url;
    }

    public async Task<string> CreatePortalSessionAsync(string stripeCustomerId, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(stripeCustomerId))
        {
            throw new ArgumentException("Stripe customer ID is required.", nameof(stripeCustomerId));
        }

        var portalOptions = new Stripe.BillingPortal.SessionCreateOptions
        {
            Customer = stripeCustomerId,
            ReturnUrl = _options.SuccessUrl
        };

        var portalService = new Stripe.BillingPortal.SessionService();
        var session = await portalService.CreateAsync(portalOptions, cancellationToken: ct).ConfigureAwait(false);

        _logger.LogInformation(
            "Created Stripe Customer Portal session for customer {CustomerId}",
            stripeCustomerId);

        return session.Url;
    }

    public async Task HandleWebhookAsync(string json, string signature, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            throw new ArgumentException("Webhook JSON body is required.", nameof(json));
        }

        if (string.IsNullOrWhiteSpace(signature))
        {
            throw new ArgumentException("Stripe signature header is required.", nameof(signature));
        }

        var stripeEvent = EventUtility.ConstructEvent(json, signature, _options.WebhookSigningSecret);

        _logger.LogInformation("Processing Stripe webhook event {EventType} ({EventId})", stripeEvent.Type, stripeEvent.Id);

        switch (stripeEvent.Type)
        {
            case EventTypes.CheckoutSessionCompleted:
                await HandleCheckoutSessionCompletedAsync(stripeEvent, ct).ConfigureAwait(false);
                break;

            case EventTypes.CustomerSubscriptionUpdated:
                await HandleSubscriptionUpdatedAsync(stripeEvent, ct).ConfigureAwait(false);
                break;

            case EventTypes.CustomerSubscriptionDeleted:
                await HandleSubscriptionDeletedAsync(stripeEvent, ct).ConfigureAwait(false);
                break;

            case EventTypes.InvoicePaymentFailed:
                HandleInvoicePaymentFailed(stripeEvent);
                break;

            default:
                _logger.LogInformation("Unhandled Stripe event type: {EventType}", stripeEvent.Type);
                break;
        }
    }

    private async Task HandleCheckoutSessionCompletedAsync(Event stripeEvent, CancellationToken ct)
    {
        var session = stripeEvent.Data.Object as Session;
        if (session is null)
        {
            _logger.LogWarning("Checkout session completed event has no session object");
            return;
        }

        string stripeCustomerId = session.CustomerId ?? string.Empty;
        if (string.IsNullOrWhiteSpace(stripeCustomerId))
        {
            _logger.LogWarning("Checkout session {SessionId} has no Stripe customer ID", session.Id);
            return;
        }
        string email = session.CustomerEmail ?? session.CustomerDetails?.Email ?? string.Empty;
        if (string.IsNullOrWhiteSpace(email))
        {
            _logger.LogWarning(
                "Checkout session {SessionId} has no customer email. An empty email will be persisted.",
                session.Id);
        }
        string customerName = session.Metadata?.GetValueOrDefault("customer_name") ?? "Unknown";
        string tierStr = session.Metadata?.GetValueOrDefault("tier") ?? "Pro";
        string seatsStr = session.Metadata?.GetValueOrDefault("seats") ?? "1";

        if (!Enum.TryParse<SubscriptionTier>(tierStr, ignoreCase: true, out var tier))
        {
            tier = SubscriptionTier.Pro;
        }

        if (!int.TryParse(seatsStr, out int seats) || seats < 1)
        {
            seats = 1;
        }

        // Upsert Stripe customer
        var customer = await _db.StripeCustomers
            .SingleOrDefaultAsync(c => c.StripeCustomerId == stripeCustomerId, ct)
            .ConfigureAwait(false);

        // ClientReferenceId carries the authenticated user ID from checkout session creation.
        string? checkoutUserId = session.ClientReferenceId;

        if (customer is null)
        {
            customer = new StripeCustomerEntity
            {
                Id = Guid.NewGuid(),
                StripeCustomerId = stripeCustomerId,
                Email = email,
                Name = customerName,
                UserId = checkoutUserId,
                CreatedAt = DateTimeOffset.UtcNow
            };
            _db.StripeCustomers.Add(customer);
        }
        else
        {
            customer.Email = email;
            customer.Name = customerName;
            if (!string.IsNullOrWhiteSpace(checkoutUserId))
            {
                customer.UserId = checkoutUserId;
            }
        }

        // Auto-create license key
        var createRequest = new LicenseCreateRequest
        {
            CustomerName = customerName,
            CustomerEmail = email,
            Tier = tier,
            MaxSeats = seats,
            Notes = $"Auto-created via Stripe checkout session {session.Id}"
        };

        var licenseResponse = await _licenseService.CreateAsync(createRequest, ct).ConfigureAwait(false);

        // Store subscription mapping (null for one-time payments — skip subscription record)
        string? subscriptionId = session.SubscriptionId;
        if (!string.IsNullOrWhiteSpace(subscriptionId))
        {
            var subscription = new StripeSubscriptionEntity
            {
                Id = Guid.NewGuid(),
                StripeSubscriptionId = subscriptionId,
                StripeCustomerId = stripeCustomerId,
                LicenseKeyId = licenseResponse.Id,
                StripePriceId = string.Empty, // Will be updated by subscription.updated event
                Status = StripeSubscriptionStatus.Active,
                // Approximate period bounds — the subscription.updated webhook event
                // (fired immediately after checkout) will correct these to actual Stripe values.
                CurrentPeriodStart = DateTimeOffset.UtcNow,
                CurrentPeriodEnd = DateTimeOffset.UtcNow.AddMonths(1),
                CreatedAt = DateTimeOffset.UtcNow
            };
            _db.StripeSubscriptions.Add(subscription);
        }

        await _db.SaveChangesAsync(ct).ConfigureAwait(false);

        _logger.LogInformation(
            "Checkout completed: created license {LicenseId} for customer {CustomerId} (Tier={Tier}, Seats={Seats})",
            licenseResponse.Id, stripeCustomerId, tier, seats);
    }

    private async Task HandleSubscriptionUpdatedAsync(Event stripeEvent, CancellationToken ct)
    {
        var subscription = stripeEvent.Data.Object as Stripe.Subscription;
        if (subscription is null)
        {
            _logger.LogWarning("Subscription updated event has no subscription object");
            return;
        }

        var entity = await _db.StripeSubscriptions
            .SingleOrDefaultAsync(s => s.StripeSubscriptionId == subscription.Id, ct)
            .ConfigureAwait(false);

        if (entity is null)
        {
            _logger.LogWarning("Subscription {SubscriptionId} not found in database", subscription.Id);
            return;
        }

        entity.Status = MapStripeStatus(subscription.Status);

        // In Stripe.net v48+, CurrentPeriodStart/End moved from Subscription to SubscriptionItem level.
        if (subscription.Items?.Data?.Count > 0)
        {
            var firstItem = subscription.Items.Data[0];
            entity.CurrentPeriodStart = firstItem.CurrentPeriodStart;
            entity.CurrentPeriodEnd = firstItem.CurrentPeriodEnd;
            entity.StripePriceId = firstItem.Price?.Id ?? entity.StripePriceId;
        }

        if (subscription.CanceledAt is not null)
        {
            entity.CancelledAt = subscription.CanceledAt;
        }

        await _db.SaveChangesAsync(ct).ConfigureAwait(false);

        _logger.LogInformation(
            "Subscription {SubscriptionId} updated to status {Status}",
            subscription.Id, entity.Status);
    }

    private async Task HandleSubscriptionDeletedAsync(Event stripeEvent, CancellationToken ct)
    {
        var subscription = stripeEvent.Data.Object as Stripe.Subscription;
        if (subscription is null)
        {
            _logger.LogWarning("Subscription deleted event has no subscription object");
            return;
        }

        var entity = await _db.StripeSubscriptions
            .SingleOrDefaultAsync(s => s.StripeSubscriptionId == subscription.Id, ct)
            .ConfigureAwait(false);

        if (entity is null)
        {
            _logger.LogWarning("Subscription {SubscriptionId} not found in database for deletion", subscription.Id);
            return;
        }

        entity.Status = StripeSubscriptionStatus.Cancelled;
        entity.CancelledAt = DateTimeOffset.UtcNow;

        // Revoke the associated license
        if (entity.LicenseKeyId.HasValue)
        {
            bool revoked = await _licenseService.RevokeAsync(entity.LicenseKeyId.Value, ct).ConfigureAwait(false);
            if (revoked)
            {
                _logger.LogInformation(
                    "Revoked license {LicenseId} due to subscription {SubscriptionId} cancellation",
                    entity.LicenseKeyId.Value, subscription.Id);
            }
        }

        await _db.SaveChangesAsync(ct).ConfigureAwait(false);

        _logger.LogInformation("Subscription {SubscriptionId} deleted/cancelled", subscription.Id);
    }

    private void HandleInvoicePaymentFailed(Event stripeEvent)
    {
        var invoice = stripeEvent.Data.Object as Invoice;
        if (invoice is null)
        {
            _logger.LogWarning("Invoice payment failed event has no invoice object");
            return;
        }

        _logger.LogWarning(
            "Invoice payment failed for customer {CustomerId}, subscription {SubscriptionId}, amount {Amount} {Currency}",
            invoice.CustomerId, invoice.Parent?.SubscriptionDetails?.SubscriptionId, invoice.AmountDue, invoice.Currency);
    }

    private string ResolvePriceId(SubscriptionTier tier, string? billingInterval)
    {
        bool isAnnual = string.Equals(billingInterval, "year", StringComparison.OrdinalIgnoreCase);

        return tier switch
        {
            SubscriptionTier.Pro when isAnnual => _options.ProAnnualPriceId,
            SubscriptionTier.Pro => _options.ProPriceId,
            SubscriptionTier.Enterprise => _options.EnterprisePriceId,
            _ => _options.ProPriceId
        };
    }

    private StripeSubscriptionStatus MapStripeStatus(string stripeStatus)
    {
        return stripeStatus switch
        {
            "active" => StripeSubscriptionStatus.Active,
            "past_due" => StripeSubscriptionStatus.PastDue,
            "canceled" => StripeSubscriptionStatus.Cancelled,
            "incomplete" => StripeSubscriptionStatus.Incomplete,
            "trialing" => StripeSubscriptionStatus.Trialing,
            "paused" => StripeSubscriptionStatus.Paused,
            _ => LogAndReturnUnknown(stripeStatus)
        };
    }

    public async Task<bool> ValidateCustomerOwnershipAsync(string userId, string stripeCustomerId, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(userId) || string.IsNullOrWhiteSpace(stripeCustomerId))
        {
            return false;
        }

        // Look up the Stripe customer in our database and verify it belongs to the requesting user.
        var customer = await _db.StripeCustomers
            .AsNoTracking()
            .SingleOrDefaultAsync(c => c.StripeCustomerId == stripeCustomerId, ct)
            .ConfigureAwait(false);

        if (customer is null)
        {
            _logger.LogWarning(
                "ValidateCustomerOwnership: Stripe customer {CustomerId} not found in database",
                stripeCustomerId);
            return false;
        }

        // Match the JWT sub claim (userId) against the stored UserId on the Stripe customer record.
        if (!string.Equals(customer.UserId, userId, StringComparison.Ordinal))
        {
            _logger.LogWarning(
                "ValidateCustomerOwnership: UserId mismatch for customer {CustomerId}",
                stripeCustomerId);
            return false;
        }

        // Verify the customer has at least one active subscription.
        bool hasSubscription = await _db.StripeSubscriptions
            .AsNoTracking()
            .AnyAsync(s => s.StripeCustomerId == stripeCustomerId
                && s.Status != StripeSubscriptionStatus.Cancelled, ct)
            .ConfigureAwait(false);

        if (!hasSubscription)
        {
            _logger.LogWarning(
                "ValidateCustomerOwnership: No active subscription found for customer {CustomerId}",
                stripeCustomerId);
        }

        return hasSubscription;
    }

    private StripeSubscriptionStatus LogAndReturnUnknown(string stripeStatus)
    {
        _logger.LogWarning(
            "Unknown Stripe subscription status '{StripeStatus}'. Treating as Unknown instead of Active. " +
            "Update MapStripeStatus to handle this status explicitly.",
            stripeStatus);
        return StripeSubscriptionStatus.Unknown;
    }
}
