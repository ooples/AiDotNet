using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Stripe;
using Xunit;

namespace AiDotNet.Playground.Functions.Tests;

/// <summary>
/// Drives the real <see cref="StripeWebhook"/> end to end with a correctly signed Stripe event and a
/// loopback fake of the Supabase admin/REST API, then inspects everything the function logged.
/// </summary>
/// <remarks>
/// The tests mutate process-wide environment variables the function reads, so they live in one class
/// (xUnit runs a class's tests sequentially) and restore the previous values afterwards.
/// </remarks>
public sealed class StripeWebhookLoggingTests : IAsyncLifetime
{
    private const string WebhookSecret = "whsec_test_do_not_leak_0123456789";
    private const string SecondarySecret = "whsec_secondary_do_not_leak_456789";
    private const string SupabaseKey = "sb_secret_test_do_not_leak_9876543210";
    private const string CustomerEmail = "jane.private@janes-family-domain.example";
    private const string CustomerId = "cus_TEST123";

    private static readonly string[] EnvNames = ["STRIPE_WEBHOOK_SECRET", "STRIPE_WEBHOOK_SECRET_SECONDARY", "STRIPE_SECRET_KEY", "SUPABASE_URL", "SUPABASE_SECRET_KEY"];

    private readonly Dictionary<string, string?> _savedEnv = new();
    private readonly ConcurrentQueue<string> _supabaseRequests = new();
    private WebApplication? _fakeSupabase;
    private string _supabaseUsersJson = "{\"users\":[]}";
    private IStripeClient? _savedStripeClient;
    private string? _savedStripeKey;
    private int _stripeStatus = 200;
    private string? _stripeEmail = CustomerEmail;

    public async Task InitializeAsync()
    {
        foreach (var name in EnvNames)
        {
            _savedEnv[name] = Environment.GetEnvironmentVariable(name);
        }

        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseSetting("urls", "http://127.0.0.1:0");
        _fakeSupabase = builder.Build();
        _fakeSupabase.MapGet("/v1/customers/{id}", (string id) => _stripeStatus == 200
            ? Results.Json(new { id, @object = "customer", email = _stripeEmail })
            : Results.Json(new { error = new { type = "api_error", message = "Temporary test failure" } }, statusCode: _stripeStatus));
        _fakeSupabase.MapGet("/auth/v1/admin/users", () =>
        {
            _supabaseRequests.Enqueue("GET users");
            return Results.Content(_supabaseUsersJson, "application/json");
        });
        _fakeSupabase.MapMethods("/rest/v1/profiles", ["PATCH"], async (HttpRequest request) =>
        {
            using var reader = new StreamReader(request.Body);
            _supabaseRequests.Enqueue("PATCH " + request.QueryString + " " + await reader.ReadToEndAsync());
            return Results.NoContent();
        });
        await _fakeSupabase.StartAsync();

        var address = _fakeSupabase.Services.GetRequiredService<IServer>()
            .Features.Get<IServerAddressesFeature>()
            ?.Addresses.First()
            ?? throw new InvalidOperationException("The fake Supabase server exposes no address.");

        Environment.SetEnvironmentVariable("STRIPE_WEBHOOK_SECRET", WebhookSecret);
        Environment.SetEnvironmentVariable("STRIPE_WEBHOOK_SECRET_SECONDARY", null);
        Environment.SetEnvironmentVariable("SUPABASE_URL", address.TrimEnd('/'));
        Environment.SetEnvironmentVariable("SUPABASE_SECRET_KEY", SupabaseKey);
        _savedStripeKey = StripeConfiguration.ApiKey;
        _savedStripeClient = StripeConfiguration.StripeClient;
        Environment.SetEnvironmentVariable("STRIPE_SECRET_KEY", "sk_test_local_only");
        StripeConfiguration.ApiKey = "sk_test_local_only";
        StripeConfiguration.StripeClient = new StripeClient("sk_test_local_only", apiBase: address.TrimEnd('/'),
            httpClient: new SystemNetHttpClient(maxNetworkRetries: 0));
    }

    public async Task DisposeAsync()
    {
        StripeConfiguration.ApiKey = _savedStripeKey;
        StripeConfiguration.StripeClient = _savedStripeClient;
        foreach (var (name, value) in _savedEnv)
        {
            Environment.SetEnvironmentVariable(name, value);
        }

        if (_fakeSupabase is not null)
        {
            await _fakeSupabase.DisposeAsync();
        }
    }

    [Fact(Timeout = 60000)]
    public async Task CheckoutCompleted_UpdatesProfile_WithoutLoggingEmailDerivedData()
    {
        _supabaseUsersJson = JsonSerializer.Serialize(new
        {
            users = new[] { new { id = "00000000-0000-0000-0000-00000000abcd", email = CustomerEmail } }
        });
        var logger = new CapturingLogger<StripeWebhook>();

        var result = await new StripeWebhook(logger).Run(SignedRequest(CheckoutCompletedEvent()));

        Assert.IsType<OkResult>(result);
        Assert.Contains(_supabaseRequests, r => r.StartsWith("PATCH ?id=eq.00000000-0000-0000-0000-00000000abcd", StringComparison.Ordinal)
                                                 && r.Contains(CustomerId, StringComparison.Ordinal));
        Assert.Contains(logger.Entries, e => e.Contains(CustomerId, StringComparison.Ordinal));
        AssertNoSensitiveData(logger.Entries);
    }

    [Fact(Timeout = 60000)]
    public async Task CheckoutCompleted_WithNoMatchingSupabaseUser_DoesNotLogEmailDerivedData()
    {
        _supabaseUsersJson = "{\"users\":[]}";
        var logger = new CapturingLogger<StripeWebhook>();

        var result = await new StripeWebhook(logger).Run(SignedRequest(CheckoutCompletedEvent()));

        Assert.IsType<OkResult>(result);
        Assert.DoesNotContain(_supabaseRequests, r => r.StartsWith("PATCH", StringComparison.Ordinal));
        Assert.Contains(logger.Entries, e => e.Contains("No Supabase user found", StringComparison.Ordinal)
                                             && e.Contains(CustomerId, StringComparison.Ordinal));
        AssertNoSensitiveData(logger.Entries);
    }

    [Theory(Timeout = 60000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task InvalidSignature_Rejected_WithoutEchoingSecretsOrPayload(bool legacy)
    {
        var logger = new CapturingLogger<StripeWebhook>();
        var payload = CheckoutCompletedEvent();
        if (legacy) payload = payload.Replace(StripeConfiguration.ApiVersion, "2026-02-25.clover", StringComparison.Ordinal);
        var request = SignedRequest(payload, signingSecret: "whsec_attacker_guess");

        var result = await new StripeWebhook(logger).Run(request);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var responseBody = JsonSerializer.Serialize(badRequest.Value);
        Assert.Equal("\"Invalid signature\"", responseBody);
        Assert.DoesNotContain(WebhookSecret, responseBody, StringComparison.Ordinal);
        Assert.Empty(_supabaseRequests);
        AssertNoSensitiveData(logger.Entries);
    }

    [Theory(Timeout = 60000)]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("malformed-signature")]
    public async Task MissingOrMalformedSignature_IsRejectedWithoutDatabaseRequests(string? signature)
    {
        var request = SignedRequest(CheckoutCompletedEvent());
        request.Headers.Remove("Stripe-Signature");
        if (signature is not null) request.Headers["Stripe-Signature"] = signature;

        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>()).Run(request);

        Assert.IsType<BadRequestObjectResult>(result);
        Assert.Empty(_supabaseRequests);
    }

    [Fact(Timeout = 60000)]
    public async Task MissingWebhookConfiguration_ReturnsRetryableFailure()
    {
        Environment.SetEnvironmentVariable("STRIPE_WEBHOOK_SECRET", null);
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>())
            .Run(SignedRequest(CheckoutCompletedEvent()));

        Assert.Equal(StatusCodes.Status500InternalServerError, Assert.IsType<StatusCodeResult>(result).StatusCode);
        Assert.Empty(_supabaseRequests);
    }

    [Fact(Timeout = 60000)]
    public async Task SignedUnhandledEvent_IsAcknowledgedWithoutDatabaseRequests()
    {
        var payload = CheckoutCompletedEvent().Replace(EventTypes.CheckoutSessionCompleted, EventTypes.InvoiceCreated, StringComparison.Ordinal);
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>()).Run(SignedRequest(payload));

        Assert.IsType<OkResult>(result);
        Assert.Empty(_supabaseRequests);
    }

    [Fact(Timeout = 60000)]
    public async Task ExistingCloverCheckout_UpdatesProfileWithSignatureVerification()
    {
        _supabaseUsersJson = JsonSerializer.Serialize(new
        {
            users = new[] { new { id = "00000000-0000-0000-0000-00000000abcd", email = CustomerEmail } }
        });
        var payload = CheckoutCompletedEvent().Replace(StripeConfiguration.ApiVersion, "2026-02-25.clover", StringComparison.Ordinal);
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>()).Run(SignedRequest(payload));
        Assert.IsType<OkResult>(result);
        AssertProfilePatch("pro", "active");
    }

    [Theory(Timeout = 60000)]
    [InlineData(EventTypes.CustomerSubscriptionUpdated, true)]
    [InlineData(EventTypes.CustomerSubscriptionUpdated, false)]
    [InlineData(EventTypes.CustomerSubscriptionDeleted, true)]
    [InlineData(EventTypes.CustomerSubscriptionDeleted, false)]
    public async Task CustomerLookupFailure_IsRetryable(string eventType, bool missingKey)
    {
        if (missingKey) Environment.SetEnvironmentVariable("STRIPE_SECRET_KEY", null);
        else _stripeStatus = 503;
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>())
            .Run(SignedRequest(SubscriptionEvent(eventType, "active")));
        Assert.Equal(500, Assert.IsType<StatusCodeResult>(result).StatusCode);
        Assert.Empty(_supabaseRequests);
    }

    [Theory(Timeout = 60000)]
    [InlineData(EventTypes.CustomerSubscriptionUpdated)]
    [InlineData(EventTypes.CustomerSubscriptionDeleted)]
    public async Task CustomerWithoutEmail_IsAcknowledgedWithoutProfileUpdate(string eventType)
    {
        _stripeEmail = null;
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>())
            .Run(SignedRequest(SubscriptionEvent(eventType, "active")));
        Assert.IsType<OkResult>(result);
        Assert.Empty(_supabaseRequests);
    }

    [Theory(Timeout = 60000)]
    [InlineData(EventTypes.CustomerSubscriptionUpdated, "past_due", "pro")]
    [InlineData(EventTypes.CustomerSubscriptionDeleted, "canceled", "free")]
    public async Task CloverSubscription_UpdatesExpectedProfile(string eventType, string status, string tier)
    {
        _supabaseUsersJson = JsonSerializer.Serialize(new
        {
            users = new[] { new { id = "00000000-0000-0000-0000-00000000abcd", email = CustomerEmail } }
        });
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>())
            .Run(SignedRequest(SubscriptionEvent(eventType, status)));
        Assert.IsType<OkResult>(result);
        AssertProfilePatch(tier, status);
    }

    [Theory(Timeout = 60000)]
    [InlineData(WebhookSecret)]
    [InlineData(SecondarySecret)]
    public async Task BothEndpointSecrets_UpdateTheExactProfile(string signingSecret)
    {
        Environment.SetEnvironmentVariable("STRIPE_WEBHOOK_SECRET_SECONDARY", SecondarySecret);
        _supabaseUsersJson = JsonSerializer.Serialize(new
        {
            users = new[] { new { id = "00000000-0000-0000-0000-00000000abcd", email = CustomerEmail } }
        });
        var logger = new CapturingLogger<StripeWebhook>();
        var payload = CheckoutCompletedEvent().Replace(StripeConfiguration.ApiVersion, "2026-02-25.clover", StringComparison.Ordinal);
        var result = await new StripeWebhook(logger).Run(SignedRequest(payload, signingSecret));
        Assert.IsType<OkResult>(result);
        AssertProfilePatch("pro", "active");
        AssertNoSensitiveData(logger.Entries);
    }

    [Theory(Timeout = 60000)]
    [InlineData("unknown-secret")]
    [InlineData("expired")]
    [InlineData("unsupported-version")]
    public async Task SecondarySecret_PreservesAuthenticationAndVersionChecks(string failure)
    {
        Environment.SetEnvironmentVariable("STRIPE_WEBHOOK_SECRET_SECONDARY", SecondarySecret);
        var payload = CheckoutCompletedEvent().Replace(StripeConfiguration.ApiVersion,
            failure == "unsupported-version" ? "2099-01-01.future" : "2026-02-25.clover", StringComparison.Ordinal);
        var request = SignedRequest(payload,
            failure == "unknown-secret" ? "whsec_untrusted" : SecondarySecret,
            failure == "expired" ? DateTimeOffset.UtcNow.AddHours(-1).ToUnixTimeSeconds() : null);
        var logger = new CapturingLogger<StripeWebhook>();
        Assert.IsType<BadRequestObjectResult>(await new StripeWebhook(logger).Run(request));
        Assert.Empty(_supabaseRequests);
        AssertNoSensitiveData(logger.Entries);
    }

    private void AssertProfilePatch(string tier, string status)
    {
        var patch = Assert.Single(_supabaseRequests, request => request.StartsWith("PATCH", StringComparison.Ordinal));
        const string prefix = "PATCH ?id=eq.00000000-0000-0000-0000-00000000abcd ";
        Assert.StartsWith(prefix, patch, StringComparison.Ordinal);
        using var document = JsonDocument.Parse(patch[prefix.Length..]);
        Assert.Equal(CustomerId, document.RootElement.GetProperty("stripe_customer_id").GetString());
        Assert.Equal(tier, document.RootElement.GetProperty("subscription_tier").GetString());
        Assert.Equal(status, document.RootElement.GetProperty("subscription_status").GetString());
    }

    private static string SubscriptionEvent(string eventType, string status) => JsonSerializer.Serialize(new
    {
        id = "evt_legacy_subscription", @object = "event", api_version = "2026-02-25.clover", type = eventType,
        data = new { @object = new { id = "sub_legacy", @object = "subscription", customer = CustomerId, status } }
    });

    [Theory(Timeout = 60000)]
    [InlineData("2025-01-27.acacia")]
    [InlineData("2099-01-01.future")]
    public async Task OtherApiVersions_AreRejectedWithoutDatabaseRequests(string version)
    {
        var payload = CheckoutCompletedEvent().Replace(StripeConfiguration.ApiVersion, version, StringComparison.Ordinal);
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>()).Run(SignedRequest(payload));
        Assert.Equal("Unsupported webhook API version", Assert.IsType<BadRequestObjectResult>(result).Value);
        Assert.Empty(_supabaseRequests);
    }

    [Theory]
    [InlineData(EventTypes.CustomerSubscriptionUpdated, "past_due")]
    [InlineData(EventTypes.CustomerSubscriptionDeleted, "canceled")]
    public void CloverSubscriptionSnapshots_PreserveFieldsUsedByTheHandler(string eventType, string status)
    {
        var payload = JsonSerializer.Serialize(new
        {
            id = "evt_legacy_subscription", @object = "event", api_version = "2026-02-25.clover", type = eventType,
            data = new { @object = new { id = "sub_legacy", @object = "subscription", customer = CustomerId, status } }
        });
        var request = SignedRequest(payload);
        var parsed = EventUtility.ConstructEvent(payload, request.Headers["Stripe-Signature"], WebhookSecret, throwOnApiVersionMismatch: false);
        var subscription = Assert.IsType<Subscription>(parsed.Data.Object);
        Assert.Equal(CustomerId, subscription.CustomerId);
        Assert.Equal(status, subscription.Status);
        Assert.Equal(eventType, parsed.Type);
    }

    [Fact(Timeout = 60000)]
    public async Task ExpiredCloverSignature_IsRejectedWithoutDatabaseRequests()
    {
        var payload = CheckoutCompletedEvent().Replace(StripeConfiguration.ApiVersion, "2026-02-25.clover", StringComparison.Ordinal);
        var request = SignedRequest(payload, timestampOverride: DateTimeOffset.UtcNow.AddHours(-1).ToUnixTimeSeconds());
        var result = await new StripeWebhook(new CapturingLogger<StripeWebhook>()).Run(request);
        Assert.IsType<BadRequestObjectResult>(result);
        Assert.Empty(_supabaseRequests);
    }

    private static void AssertNoSensitiveData(IReadOnlyCollection<string> entries)
    {
        Assert.NotEmpty(entries);
        foreach (var entry in entries)
        {
            // Not the address, not a masked form of it (j***@domain), not the domain on its own.
            Assert.DoesNotContain("@", entry, StringComparison.Ordinal);
            Assert.DoesNotContain("jane.private", entry, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("janes-family-domain", entry, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(WebhookSecret, entry, StringComparison.Ordinal);
            Assert.DoesNotContain(SecondarySecret, entry, StringComparison.Ordinal);
            Assert.DoesNotContain(SupabaseKey, entry, StringComparison.Ordinal);
        }
    }

    private static string CheckoutCompletedEvent() => JsonSerializer.Serialize(new Dictionary<string, object?>
    {
        ["id"] = "evt_test_checkout",
        ["object"] = "event",
        ["api_version"] = StripeConfiguration.ApiVersion,
        ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        ["livemode"] = false,
        ["pending_webhooks"] = 1,
        ["type"] = EventTypes.CheckoutSessionCompleted,
        ["request"] = new Dictionary<string, object?> { ["id"] = null, ["idempotency_key"] = null },
        ["data"] = new Dictionary<string, object?>
        {
            ["object"] = new Dictionary<string, object?>
            {
                ["id"] = "cs_test_session",
                ["object"] = "checkout.session",
                ["mode"] = "subscription",
                ["status"] = "complete",
                ["customer"] = CustomerId,
                ["customer_email"] = CustomerEmail,
                ["customer_details"] = new Dictionary<string, object?> { ["email"] = CustomerEmail },
                ["subscription"] = "sub_TEST456"
            }
        }
    });

    private static HttpRequest SignedRequest(string payload, string signingSecret = WebhookSecret, long? timestampOverride = null)
    {
        var timestamp = timestampOverride ?? DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(signingSecret));
        var signature = Convert.ToHexString(hmac.ComputeHash(Encoding.UTF8.GetBytes($"{timestamp}.{payload}")))
            .ToLowerInvariant();

        var context = new DefaultHttpContext();
        context.Request.Method = "POST";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(payload));
        context.Request.Headers["Stripe-Signature"] = $"t={timestamp},v1={signature}";
        return context.Request;
    }

    /// <summary>Records every rendered message, structured value and exception the function logs.</summary>
    private sealed class CapturingLogger<T> : ILogger<T>
    {
        private readonly ConcurrentQueue<string> _entries = new();

        public IReadOnlyCollection<string> Entries => _entries.ToArray();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            var values = state is IEnumerable<KeyValuePair<string, object?>> pairs
                ? string.Join(" | ", pairs.Select(p => $"{p.Key}={p.Value}"))
                : string.Empty;
            _entries.Enqueue($"{logLevel}: {formatter(state, exception)} [{values}] {exception}");
        }
    }
}
