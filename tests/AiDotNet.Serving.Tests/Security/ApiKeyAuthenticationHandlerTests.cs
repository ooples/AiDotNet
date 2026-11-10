using System.Text.Encodings.Web;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.Security;

/// <summary>
/// Credential-extraction order of the real <see cref="ApiKeyAuthenticationHandler"/>: the configured header,
/// then the bearer credential, then the Anthropic/Claude Code <c>x-api-key</c> header.
/// </summary>
public sealed class ApiKeyAuthenticationHandlerTests
{
    private const string ValidKey = "aidn_valid_key";

    [Fact]
    public async Task XApiKey_Alone_Authenticates()
    {
        var (result, presented) = await AuthenticateAsync(ctx => ctx.Request.Headers["x-api-key"] = ValidKey);

        Assert.True(result.Succeeded);
        Assert.Equal(new[] { ValidKey }, presented);
    }

    [Fact]
    public async Task Bearer_TakesPrecedenceOver_UnrelatedXApiKey()
    {
        var (result, presented) = await AuthenticateAsync(ctx =>
        {
            ctx.Request.Headers["Authorization"] = "Bearer " + ValidKey;
            ctx.Request.Headers["x-api-key"] = "sk-ant-upstream-key-not-for-aidotnet";
        });

        Assert.True(result.Succeeded);
        Assert.Equal(new[] { ValidKey }, presented);
    }

    [Fact]
    public async Task ConfiguredHeader_TakesPrecedenceOver_BearerAndXApiKey()
    {
        var (result, presented) = await AuthenticateAsync(ctx =>
        {
            ctx.Request.Headers["X-AiDotNet-ApiKey"] = ValidKey;
            ctx.Request.Headers["Authorization"] = "Bearer other";
            ctx.Request.Headers["x-api-key"] = "another";
        });

        Assert.True(result.Succeeded);
        Assert.Equal(new[] { ValidKey }, presented);
    }

    [Fact]
    public async Task XApiKey_TakesPrecedenceOver_QueryString()
    {
        var (result, presented) = await AuthenticateAsync(ctx =>
        {
            ctx.Request.Headers["x-api-key"] = ValidKey;
            ctx.Request.QueryString = new QueryString("?api_key=from-the-query");
        }, new ApiKeyAuthenticationOptions { AllowQueryString = true });

        Assert.True(result.Succeeded);
        Assert.Equal(new[] { ValidKey }, presented);
    }

    [Fact]
    public async Task WhitespaceXApiKey_FallsThroughTo_QueryString()
    {
        var (result, presented) = await AuthenticateAsync(ctx =>
        {
            ctx.Request.Headers["x-api-key"] = "   ";
            ctx.Request.QueryString = new QueryString("?api_key=" + ValidKey);
        }, new ApiKeyAuthenticationOptions { AllowQueryString = true });

        Assert.True(result.Succeeded);
        Assert.Equal(new[] { ValidKey }, presented);
    }

    private static async Task<(AuthenticateResult Result, List<string> Presented)> AuthenticateAsync(Action<HttpContext> arrange,
        ApiKeyAuthenticationOptions? options = null)
    {
        var service = new RecordingApiKeyService();
        var handler = new ApiKeyAuthenticationHandler(
            new StaticOptionsMonitor(options ?? new ApiKeyAuthenticationOptions()),
            NullLoggerFactory.Instance,
            UrlEncoder.Default,
            service);
        var context = new DefaultHttpContext();
        arrange(context);
        await handler.InitializeAsync(
            new AuthenticationScheme(ApiKeyAuthenticationDefaults.Scheme, null, typeof(ApiKeyAuthenticationHandler)),
            context);
        var result = await handler.AuthenticateAsync();
        return (result, service.Presented);
    }

    private sealed class RecordingApiKeyService : IApiKeyService
    {
        public List<string> Presented { get; } = new List<string>();

        public Task<ApiKeyAuthenticationResult?> AuthenticateAsync(string apiKey, CancellationToken cancellationToken = default)
        {
            Presented.Add(apiKey);
            ApiKeyAuthenticationResult? result = apiKey == ValidKey
                ? new ApiKeyAuthenticationResult("key-1", SubscriptionTier.Free, ApiKeyScopes.Inference)
                : null;
            return Task.FromResult(result);
        }

        public Task<ApiKeyCreateResponse> CreateAsync(ApiKeyCreateRequest request, CancellationToken cancellationToken = default)
            => throw new NotSupportedException("Not used by the authentication handler.");

        public Task<bool> RevokeAsync(string keyId, CancellationToken cancellationToken = default)
            => throw new NotSupportedException("Not used by the authentication handler.");
    }

    private sealed class StaticOptionsMonitor : IOptionsMonitor<ApiKeyAuthenticationOptions>
    {
        public StaticOptionsMonitor(ApiKeyAuthenticationOptions value) => CurrentValue = value;

        public ApiKeyAuthenticationOptions CurrentValue { get; }

        public ApiKeyAuthenticationOptions Get(string? name) => CurrentValue;

        public IDisposable? OnChange(Action<ApiKeyAuthenticationOptions, string?> listener) => null;
    }
}
