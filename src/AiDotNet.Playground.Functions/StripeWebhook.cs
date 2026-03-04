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

    public StripeWebhook(ILogger<StripeWebhook> logger)
    {
        _logger = logger;
    }

    [Function("stripe-webhook")]
    public async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req)
    {
        // Add CORS headers
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Origin", "*");
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Methods", "POST, OPTIONS");
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Headers", "Content-Type, Stripe-Signature");

        if (req.Method == "OPTIONS")
        {
            return new OkResult();
        }

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

        Event stripeEvent;
        try
        {
            stripeEvent = EventUtility.ConstructEvent(
                requestBody,
                req.Headers["Stripe-Signature"],
                webhookSecret
            );
        }
        catch (StripeException ex)
        {
            _logger.LogWarning(ex, "Stripe webhook signature verification failed");
            return new BadRequestObjectResult("Invalid signature");
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
            // Return 200 so Stripe doesn't retry - we logged the error for debugging
            return new OkResult();
        }

        return new OkResult();
    }

    private async Task HandleCheckoutSessionCompleted(Event stripeEvent)
    {
        var session = stripeEvent.Data.Object as Stripe.Checkout.Session;
        if (session is null)
        {
            _logger.LogWarning("CheckoutSessionCompleted: could not deserialize session object");
            return;
        }

        var customerEmail = session.CustomerDetails?.Email ?? session.CustomerEmail;
        var customerId = session.CustomerId;

        if (string.IsNullOrEmpty(customerEmail))
        {
            _logger.LogWarning("CheckoutSessionCompleted: no customer email found for session {SessionId}", session.Id);
            return;
        }

        _logger.LogInformation(
            "Checkout completed for customer {CustomerId}, subscription {SubscriptionId}",
            customerId, session.SubscriptionId);

        // Determine tier from the subscription
        var tier = "pro"; // Default to pro since that's our only paid tier currently

        await UpdateSupabaseProfile(customerEmail, tier, "active", customerId);
    }

    private async Task HandleSubscriptionUpdated(Event stripeEvent)
    {
        var subscription = stripeEvent.Data.Object as Subscription;
        if (subscription is null)
        {
            _logger.LogWarning("SubscriptionUpdated: could not deserialize subscription object");
            return;
        }

        // Get customer email from Stripe
        var customerEmail = await GetCustomerEmail(subscription.CustomerId);
        if (string.IsNullOrEmpty(customerEmail))
        {
            _logger.LogWarning("SubscriptionUpdated: could not find email for customer {CustomerId}", subscription.CustomerId);
            return;
        }

        var status = subscription.Status switch
        {
            "active" => "active",
            "past_due" => "past_due",
            "canceled" => "canceled",
            "unpaid" => "unpaid",
            "trialing" => "active",
            _ => subscription.Status
        };

        var tier = status == "canceled" ? "free" : "pro";

        _logger.LogInformation(
            "Subscription updated for {Email}: status={Status}, tier={Tier}",
            customerEmail, status, tier);

        await UpdateSupabaseProfile(customerEmail, tier, status, subscription.CustomerId);
    }

    private async Task HandleSubscriptionDeleted(Event stripeEvent)
    {
        var subscription = stripeEvent.Data.Object as Subscription;
        if (subscription is null)
        {
            _logger.LogWarning("SubscriptionDeleted: could not deserialize subscription object");
            return;
        }

        var customerEmail = await GetCustomerEmail(subscription.CustomerId);
        if (string.IsNullOrEmpty(customerEmail))
        {
            _logger.LogWarning("SubscriptionDeleted: could not find email for customer {CustomerId}", subscription.CustomerId);
            return;
        }

        _logger.LogInformation("Subscription deleted for {Email}, reverting to free tier", customerEmail);

        await UpdateSupabaseProfile(customerEmail, "free", "canceled", subscription.CustomerId);
    }

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
            return null;
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
            return null;
        }
    }

    private async Task UpdateSupabaseProfile(string email, string tier, string status, string? stripeCustomerId)
    {
        var supabaseUrl = Environment.GetEnvironmentVariable("SUPABASE_URL");
        var supabaseKey = Environment.GetEnvironmentVariable("SUPABASE_SECRET_KEY");

        if (string.IsNullOrEmpty(supabaseUrl) || string.IsNullOrEmpty(supabaseKey))
        {
            _logger.LogError("SUPABASE_URL or SUPABASE_SECRET_KEY environment variables are not set");
            return;
        }

        // First, find the user by email in Supabase Auth
        using var httpClient = new HttpClient();
        httpClient.DefaultRequestHeaders.Add("apikey", supabaseKey);
        httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {supabaseKey}");

        // Look up user by email via auth admin API
        var usersResponse = await httpClient.GetAsync($"{supabaseUrl}/auth/v1/admin/users");
        if (!usersResponse.IsSuccessStatusCode)
        {
            _logger.LogError("Failed to list Supabase users: {StatusCode}", usersResponse.StatusCode);
            return;
        }

        var usersJson = await usersResponse.Content.ReadAsStringAsync();
        var usersData = JsonDocument.Parse(usersJson);

        string? userId = null;
        if (usersData.RootElement.TryGetProperty("users", out var usersArray))
        {
            foreach (var user in usersArray.EnumerateArray())
            {
                if (user.TryGetProperty("email", out var emailProp) &&
                    string.Equals(emailProp.GetString(), email, StringComparison.OrdinalIgnoreCase))
                {
                    userId = user.GetProperty("id").GetString();
                    break;
                }
            }
        }

        if (string.IsNullOrEmpty(userId))
        {
            _logger.LogWarning("No Supabase user found with email {Email}", email);
            return;
        }

        // Update the profiles table via REST API
        var updatePayload = new Dictionary<string, object?>
        {
            ["subscription_tier"] = tier,
            ["subscription_status"] = status,
            ["updated_at"] = DateTime.UtcNow.ToString("o")
        };

        if (!string.IsNullOrEmpty(stripeCustomerId))
        {
            updatePayload["stripe_customer_id"] = stripeCustomerId;
        }

        var jsonContent = new StringContent(
            JsonSerializer.Serialize(updatePayload),
            Encoding.UTF8,
            "application/json"
        );

        // PATCH the profile row matching this user's ID
        var patchRequest = new HttpRequestMessage(HttpMethod.Patch,
            $"{supabaseUrl}/rest/v1/profiles?id=eq.{userId}")
        {
            Content = jsonContent
        };
        patchRequest.Headers.Add("apikey", supabaseKey);
        patchRequest.Headers.Add("Authorization", $"Bearer {supabaseKey}");
        patchRequest.Headers.Add("Prefer", "return=minimal");

        var patchResponse = await httpClient.SendAsync(patchRequest);

        if (patchResponse.IsSuccessStatusCode)
        {
            _logger.LogInformation(
                "Updated profile for user {UserId} ({Email}): tier={Tier}, status={Status}, stripe_customer={CustomerId}",
                userId, email, tier, status, stripeCustomerId);
        }
        else
        {
            var errorBody = await patchResponse.Content.ReadAsStringAsync();
            _logger.LogError(
                "Failed to update profile for user {UserId}: {StatusCode} - {Error}",
                userId, patchResponse.StatusCode, errorBody);
        }
    }
}
