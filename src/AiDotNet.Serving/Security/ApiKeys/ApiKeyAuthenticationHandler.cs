using System.Security.Claims;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Authentication handler for API keys.
/// </summary>
public sealed class ApiKeyAuthenticationHandler : AuthenticationHandler<ApiKeyAuthenticationOptions>
{
    private readonly IApiKeyService _apiKeys;

    public ApiKeyAuthenticationHandler(
        IOptionsMonitor<ApiKeyAuthenticationOptions> options,
        ILoggerFactory logger,
        System.Text.Encodings.Web.UrlEncoder encoder,
        ISystemClock clock,
        IApiKeyService apiKeys)
        : base(options, logger, encoder, clock)
    {
        _apiKeys = apiKeys ?? throw new ArgumentNullException(nameof(apiKeys));
    }

    protected override async Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        string? presented = ExtractApiKey();
        if (string.IsNullOrWhiteSpace(presented))
        {
            return AuthenticateResult.NoResult();
        }

        var result = await _apiKeys.AuthenticateAsync(presented, Context.RequestAborted).ConfigureAwait(false);
        if (result == null)
        {
            return AuthenticateResult.Fail("Invalid API key.");
        }

        var claims = new List<Claim>
        {
            new(ApiKeyClaimTypes.KeyId, result.KeyId),
            new(ApiKeyClaimTypes.Tier, result.Tier.ToString()),
            new(ClaimTypes.NameIdentifier, result.KeyId)
        };

        foreach (var scope in EnumerateScopes(result.Scopes))
        {
            claims.Add(new Claim(ApiKeyClaimTypes.Scope, scope));
        }

        var identity = new ClaimsIdentity(claims, ApiKeyAuthenticationDefaults.Scheme);
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, ApiKeyAuthenticationDefaults.Scheme);
        return AuthenticateResult.Success(ticket);
    }

    private string? ExtractApiKey()
    {
        if (!string.IsNullOrWhiteSpace(Options.HeaderName) &&
            Request.Headers.TryGetValue(Options.HeaderName, out var headerValues))
        {
            var value = headerValues.ToString();
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value.Trim();
            }
        }

        if (Request.Headers.TryGetValue("Authorization", out var authValues))
        {
            var raw = authValues.ToString();
            const string Bearer = "Bearer ";
            if (!string.IsNullOrWhiteSpace(raw) && raw.StartsWith(Bearer, StringComparison.OrdinalIgnoreCase))
            {
                return raw.Substring(Bearer.Length).Trim();
            }
        }

        if (Options.AllowQueryString &&
            !string.IsNullOrWhiteSpace(Options.QueryStringParameterName) &&
            Request.Query.TryGetValue(Options.QueryStringParameterName, out var queryValues))
        {
            var value = queryValues.ToString();
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value.Trim();
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumerateScopes(ApiKeyScopes scopes)
    {
        if (scopes == ApiKeyScopes.None)
        {
            yield break;
        }

        foreach (ApiKeyScopes value in Enum.GetValues(typeof(ApiKeyScopes)))
        {
            if (value == ApiKeyScopes.None || value == ApiKeyScopes.All)
            {
                continue;
            }

            if (scopes.HasFlag(value))
            {
                yield return value.ToString();
            }
        }
    }
}

