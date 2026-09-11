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
    private const string SupabaseKey = "sb_secret_test_do_not_leak_9876543210";
    private const string CustomerEmail = "jane.private@janes-family-domain.example";
    private const string CustomerId = "cus_TEST123";

    private static readonly string[] EnvNames = ["STRIPE_WEBHOOK_SECRET", "SUPABASE_URL", "SUPABASE_SECRET_KEY"];

    private readonly Dictionary<string, string?> _savedEnv = new();
    private readonly ConcurrentQueue<string> _supabaseRequests = new();
    private WebApplication? _fakeSupabase;
    private string _supabaseUsersJson = "{\"users\":[]}";

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
            .Features.Get<IServerAddressesFeature>()!.Addresses.First();

        Environment.SetEnvironmentVariable("STRIPE_WEBHOOK_SECRET", WebhookSecret);
        Environment.SetEnvironmentVariable("SUPABASE_URL", address.TrimEnd('/'));
        Environment.SetEnvironmentVariable("SUPABASE_SECRET_KEY", SupabaseKey);
    }

    public async Task DisposeAsync()
    {
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

    [Fact(Timeout = 60000)]
    public async Task InvalidSignature_Rejected_WithoutEchoingSecretsOrPayload()
    {
        var logger = new CapturingLogger<StripeWebhook>();
        var request = SignedRequest(CheckoutCompletedEvent(), signingSecret: "whsec_attacker_guess");

        var result = await new StripeWebhook(logger).Run(request);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var responseBody = JsonSerializer.Serialize(badRequest.Value);
        Assert.Equal("\"Invalid signature\"", responseBody);
        Assert.DoesNotContain(WebhookSecret, responseBody, StringComparison.Ordinal);
        Assert.Empty(_supabaseRequests);
        AssertNoSensitiveData(logger.Entries);
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

    private static HttpRequest SignedRequest(string payload, string signingSecret = WebhookSecret)
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
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
