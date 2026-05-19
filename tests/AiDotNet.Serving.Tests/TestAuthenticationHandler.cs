using System.Security.Claims;
using System.Text.Encodings.Web;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Test-only authentication handler that succeeds for every incoming request
/// with a fixed identity and the <see cref="ApiKeyScopes.Admin"/> scope.
/// </summary>
/// <remarks>
/// <para>
/// PR #1384 added a <c>FallbackPolicy = RequireAuthenticatedUser</c> to the
/// serving host so unprotected controllers (Inference, Embeddings, Federated,
/// Models, ProgramSynthesis/*) now require authentication. The existing
/// integration test suite drives those endpoints with a bare
/// <c>WebApplicationFactory.CreateClient()</c> — no API key in the request
/// headers — which would otherwise return 401 against the new default.
/// </para>
/// <para>
/// This handler replaces the real <see cref="ApiKeyAuthenticationHandler"/>
/// in test runs so the existing tests stay green while the production
/// host keeps its hardened defaults. Tests that specifically want to
/// verify the unauthenticated-request rejection path should NOT use this
/// handler and instead drive the host with the real auth scheme.
/// </para>
/// </remarks>
public sealed class TestAuthenticationHandler : AuthenticationHandler<ApiKeyAuthenticationOptions>
{
    public const string Scheme = "TestAuth";
    public const string TestKeyId = "test-key";

    /// <summary>
    /// Optional request header that lets a test pin the principal's
    /// <see cref="SubscriptionTier"/> claim. When present, the value is
    /// parsed (case-insensitively) and used as the Tier claim; when
    /// absent, the default is <see cref="SubscriptionTier.Enterprise"/>
    /// so tests don't need any setup for the common "do whatever, just
    /// authenticate me" case.
    /// </summary>
    public const string TestTierHeader = "X-Test-Tier";

    public TestAuthenticationHandler(
        IOptionsMonitor<ApiKeyAuthenticationOptions> options,
        ILoggerFactory logger,
        UrlEncoder encoder)
        : base(options, logger, encoder)
    {
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        // Default to Enterprise so the bulk of tests (that don't care
        // about tier policy) succeed without any per-request setup.
        // Tests that exercise tier-gated endpoints can set the
        // X-Test-Tier header to pin a specific tier.
        SubscriptionTier tier = SubscriptionTier.Enterprise;
        if (Request.Headers.TryGetValue(TestTierHeader, out var tierHeader) &&
            tierHeader.Count > 0 &&
            System.Enum.TryParse<SubscriptionTier>(tierHeader[0], ignoreCase: true, out var parsed))
        {
            tier = parsed;
        }

        var claims = new[]
        {
            new Claim(ApiKeyClaimTypes.KeyId, TestKeyId),
            new Claim(ApiKeyClaimTypes.Tier, tier.ToString()),
            new Claim(ClaimTypes.NameIdentifier, TestKeyId),
            new Claim(ApiKeyClaimTypes.Scope, ApiKeyScopes.Admin.ToString())
        };
        var identity = new ClaimsIdentity(claims, Scheme);
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, Scheme);
        return Task.FromResult(AuthenticateResult.Success(ticket));
    }
}
