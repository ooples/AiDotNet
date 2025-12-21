using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Security;

/// <summary>
/// Resolves the tier from the authenticated request principal (API key claims).
/// </summary>
public sealed class ClaimsTierResolver : ITierResolver
{
    private readonly TierEnforcementOptions _options;

    public ClaimsTierResolver(IOptions<TierEnforcementOptions> options)
    {
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
    }

    public SubscriptionTier ResolveTier(HttpContext httpContext)
    {
        if (!_options.Enabled)
        {
            return SubscriptionTier.Enterprise;
        }

        if (httpContext == null)
        {
            return _options.DefaultTier;
        }

        var user = httpContext.User;
        if (user?.Identity?.IsAuthenticated != true)
        {
            return _options.DefaultTier;
        }

        var tierClaim = user.FindFirst(ApiKeyClaimTypes.Tier)?.Value;
        if (string.IsNullOrWhiteSpace(tierClaim))
        {
            return _options.DefaultTier;
        }

        return Enum.TryParse<SubscriptionTier>(tierClaim, ignoreCase: true, out var tier)
            ? tier
            : _options.DefaultTier;
    }
}

