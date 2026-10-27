using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using Stripe;

namespace AiDotNet.Playground.Functions;

public class StripeWebhook
{
    private readonly ILogger<StripeWebhook> _logger;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    // Shared HttpClient to avoid socket exhaustion from per-request instantiation
    private static readonly HttpClient SharedHttpClient = new()
    {
        Timeout = TimeSpan.FromSeconds(30)
    };

    public StripeWebhook(ILogger<StripeWebhook> logger)
    {
        _logger = logger;
    }

    [Function("stripe-webhook")]
    public async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req)
    {
        // Stripe calls webhooks server-to-server; CORS is not needed.
        // If browser-origin calls are needed later, configure CORS in host.json instead.

        var webhookSecret = Environment.GetEnvironmentVariable("STRIPE_WEBHOOK_SECRET");
        if (string.IsNullOrEmpty(webhookSecret))
        {
            _logger.LogError("STRIPE_WEBHOOK_SECRET environment variable is not set");
            return new StatusCodeResult(StatusCodes.Status500InternalServerError);
        }

        // Read the raw body for signature verification
        string requestBody;
        using (var reader = new StreamReader(req.Body, Encoding.UTF8))
        {
            requestBody = await reader.ReadToEndAsync();
        }

        var signature = req.Headers["Stripe-Signature"].ToString();
        if (string.IsNullOrWhiteSpace(signature))
        {
            return new BadRequestObjectResult("Invalid signature");
        }

        Event stripeEvent;
        var secondarySecret = Environment.GetEnvironmentVariable("STRIPE_WEBHOOK_SECRET_SECONDARY");
        try
        {
            try
            {
                stripeEvent = EventUtility.ConstructEvent(
                    requestBody,
                    signature,
                    webhookSecret,
                    // The existing sandbox emits Clover snapshots. The three handled events
                    // use fields verified by our compatibility tests; do not accept arbitrary
                    // API versions when updating the SDK. Signature/timestamp checks remain on.
                    throwOnApiVersionMismatch: false
                );
            }
            catch (StripeException) when (!string.IsNullOrWhiteSpace(secondarySecret))
            {
                // Both existing registrations target this route with distinct signing secrets.
                // The secondary registration retains the same signature/timestamp requirements.
                stripeEvent = EventUtility.ConstructEvent(
                    requestBody, signature, secondarySecret, throwOnApiVersionMismatch: false);
            }
        }
        catch (StripeException ex)
        {
            _logger.LogWarning(ex, "Stripe webhook signature verification failed");
            return new BadRequestObjectResult("Invalid signature");
        }

        if (stripeEvent.ApiVersion != StripeConfiguration.ApiVersion && stripeEvent.ApiVersion != "2026-02-25.clover")
        {
            _logger.LogWarning("Stripe webhook uses an unsupported API version");
            return new BadRequestObjectResult("Unsupported webhook API version");
        }

        _logger.LogInformation("Received Stripe event: {EventType} ({EventId})", stripeEvent.Type, stripeEvent.Id);

        try
        {
            switch (stripeEvent.Type)
            {
                case EventTypes.CheckoutSessionCompleted:
                    await HandleCheckoutSessionCompleted(stripeEvent);
                    break;

                case EventTypes.CustomerSubscriptionUpdated:
                    await HandleSubscriptionUpdated(stripeEvent);
                    break;

                case EventTypes.CustomerSubscriptionDeleted:
                    await HandleSubscriptionDeleted(stripeEvent);
                    break;

                default:
                    _logger.LogInformation("Unhandled event type: {EventType}", stripeEvent.Type);
                    break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing Stripe event {EventType} ({EventId})", stripeEvent.Type, stripeEvent.Id);
            // Return 500 so Stripe retries - transient errors (DB down, network) should be retried.
            // Stripe retries with exponential backoff up to 72 hours.
            return new StatusCodeResult(StatusCodes.Status500InternalServerError);
        }

        return new OkResult();
    }

    private async Task HandleCheckoutSessionCompleted(Event stripeEvent)
    {
        var session = stripeEvent.Data.Object as Stripe.Checkout.Session
            ?? throw new InvalidOperationException("Checkout session could not be deserialized.");
        if (session.Mode != "subscription") return;
        var email = session.CustomerDetails?.Email ?? session.CustomerEmail;
        if (string.IsNullOrEmpty(email))
        {
            _logger.LogWarning("CheckoutSessionCompleted: no customer email for session {SessionId}", session.Id);
            return;
        }
        await UpdateSupabaseProfile(email, session.CustomerId, session.SubscriptionId, stripeEvent);
    }

    private async Task HandleSubscriptionUpdated(Event stripeEvent)
    {
        var subscription = stripeEvent.Data.Object as Subscription
            ?? throw new InvalidOperationException("Subscription could not be deserialized.");
        var email = await GetCustomerEmail(subscription.CustomerId);
        if (string.IsNullOrEmpty(email))
        {
            _logger.LogWarning("Subscription event: no email for customer {CustomerId}", subscription.CustomerId);
            return;
        }
        await UpdateSupabaseProfile(email, subscription.CustomerId, subscription.Id, stripeEvent);
    }

    private Task HandleSubscriptionDeleted(Event stripeEvent) => HandleSubscriptionUpdated(stripeEvent);

    private async Task<string?> GetCustomerEmail(string? customerId)
    {
        if (string.IsNullOrEmpty(customerId))
        {
            return null;
        }

        var stripeSecretKey = Environment.GetEnvironmentVariable("STRIPE_SECRET_KEY");
        if (string.IsNullOrEmpty(stripeSecretKey))
        {
            _logger.LogError("STRIPE_SECRET_KEY environment variable is not set");
            throw new InvalidOperationException("Stripe customer lookup configuration is missing.");
        }

        StripeConfiguration.ApiKey = stripeSecretKey;
        var customerService = new CustomerService();

        try
        {
            var customer = await customerService.GetAsync(customerId);
            return customer.Email;
        }
        catch (StripeException ex)
        {
            _logger.LogError(ex, "Failed to retrieve customer {CustomerId} from Stripe", customerId);
            throw;
        }
    }

    /// <summary>
    /// Updates the user's subscription profile in Supabase.
    /// Throws on configuration or HTTP failures so the webhook can return 500
    /// and Stripe will retry the event.
    /// </summary>
    /// <remarks>
    /// The customer's email is used only to find the Supabase user; it is never logged, not even
    /// partially (a masked "j***@domain" still leaks the domain, which identifies people on
    /// personal or small-company domains). Logs correlate through the Stripe customer ID and the
    /// Supabase user ID instead.
    /// </remarks>
    private async Task UpdateSupabaseProfile(string email, string? stripeCustomerId, string? subscriptionId, Event stripeEvent)
    {
        var supabaseUrl = Environment.GetEnvironmentVariable("SUPABASE_URL");
        var supabaseKey = Environment.GetEnvironmentVariable("SUPABASE_SECRET_KEY");

        if (string.IsNullOrEmpty(supabaseUrl) || string.IsNullOrEmpty(supabaseKey))
        {
            _logger.LogError("SUPABASE_URL or SUPABASE_SECRET_KEY environment variables are not set");
            throw new InvalidOperationException("Supabase configuration is missing. Cannot update profile.");
        }

        // First, find the user by email in Supabase Auth
        // Use per-request message headers instead of DefaultRequestHeaders to avoid thread-safety issues
        var httpClient = SharedHttpClient;

        // Look up user by email via auth admin API (paginated to avoid truncation)
        string? userId = null;
        int page = 1;
        const int perPage = 100;

        while (userId is null)
        {
            var usersRequest = new HttpRequestMessage(HttpMethod.Get,
                $"{supabaseUrl}/auth/v1/admin/users?page={page}&per_page={perPage}");
            usersRequest.Headers.Add("apikey", supabaseKey);
            usersRequest.Headers.Add("Authorization", $"Bearer {supabaseKey}");
            var usersResponse = await httpClient.SendAsync(usersRequest);
            if (!usersResponse.IsSuccessStatusCode)
            {
                _logger.LogError("Failed to list Supabase users: {StatusCode}", usersResponse.StatusCode);
                throw new InvalidOperationException($"Supabase user lookup failed with HTTP {(int)usersResponse.StatusCode}.");
            }

            var usersJson = await usersResponse.Content.ReadAsStringAsync();
            using var usersData = JsonDocument.Parse(usersJson);

            int count = 0;
            if (usersData.RootElement.TryGetProperty("users", out var usersArray))
            {
                foreach (var user in usersArray.EnumerateArray())
                {
                    count++;
                    if (user.TryGetProperty("email", out var emailProp) &&
                        string.Equals(emailProp.GetString(), email, StringComparison.OrdinalIgnoreCase))
                    {
                        userId = user.GetProperty("id").GetString();
                        break;
                    }
                }
            }

            // No more pages if fewer results than page size
            if (count < perPage)
            {
                break;
            }

            page++;
        }

        if (string.IsNullOrEmpty(userId))
        {
            _logger.LogWarning("No Supabase user found for Stripe customer {CustomerId}", stripeCustomerId);
            return;
        }

        if (string.IsNullOrEmpty(stripeCustomerId) || string.IsNullOrEmpty(subscriptionId)
            || string.IsNullOrEmpty(stripeEvent.Id) || stripeEvent.Created <= DateTime.UnixEpoch)
            throw new InvalidOperationException("Stripe event is missing its subscription or ordering identity.");

        var token = Guid.NewGuid();
        var claim = await CallStateRpc(supabaseUrl, supabaseKey, "claim_playground_stripe_event", new
        {
            p_profile_id = userId, p_event_id = stripeEvent.Id,
            p_event_created = new DateTimeOffset(stripeEvent.Created).ToUnixTimeSeconds(), p_token = token
        });
        if (claim is "duplicate" or "stale")
        {
            _logger.LogInformation("Ignored {Disposition} Stripe event {EventId}", claim, stripeEvent.Id);
            return;
        }
        if (claim != "claimed")
            throw new InvalidOperationException("Stripe profile is being processed; retry this event.");

        try
        {
            var stripeKey = Environment.GetEnvironmentVariable("STRIPE_SECRET_KEY");
            if (string.IsNullOrEmpty(stripeKey))
                throw new InvalidOperationException("Stripe subscription lookup configuration is missing.");
            StripeConfiguration.ApiKey = stripeKey;
            // Read AFTER acquiring the durable lease. Current state resolves same-second events
            // without ordering opaque IDs, and prevents delayed checkout restoring cancellation.
            var subscription = await new SubscriptionService().GetAsync(subscriptionId);
            if (subscription.CustomerId != stripeCustomerId || subscription.Id != subscriptionId)
                throw new InvalidOperationException("Stripe subscription does not match the event customer.");
            var outcome = await CallStateRpc(supabaseUrl, supabaseKey, "complete_playground_stripe_event", new
            {
                p_profile_id = userId, p_token = token, p_customer_id = stripeCustomerId,
                p_subscription_id = subscription.Id,
                p_subscription_created = new DateTimeOffset(subscription.Created).ToUnixTimeSeconds(),
                p_status = subscription.Status
            });
            if (outcome is not ("applied" or "stale"))
                throw new InvalidOperationException("Unexpected Stripe event completion response.");
            _logger.LogInformation("Stripe event {EventId} {Outcome} for user {UserId}, customer {CustomerId}",
                stripeEvent.Id, outcome, userId, stripeCustomerId);
        }
        finally
        {
            // Completion releases atomically. On failure release early; crashed workers expire.
            // Token fencing prevents an old worker releasing a replacement worker's lease.
            try
            {
                await CallStateRpc(supabaseUrl, supabaseKey, "release_playground_stripe_event",
                    new { p_profile_id = userId, p_token = token });
            }
            catch (Exception)
            {
                _logger.LogWarning("Could not release Stripe event lease; it will expire automatically");
            }
        }
    }

    private static async Task<string?> CallStateRpc(string url, string key, string function, object payload)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, $"{url}/rest/v1/rpc/{function}")
        {
            Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json")
        };
        request.Headers.Add("apikey", key);
        request.Headers.Add("Authorization", $"Bearer {key}");
        using var response = await SharedHttpClient.SendAsync(request);
        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException($"Supabase event state failed with HTTP {(int)response.StatusCode}.");
        var body = await response.Content.ReadAsStringAsync();
        return string.IsNullOrWhiteSpace(body) ? null : JsonSerializer.Deserialize<string>(body);
    }
}
