using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Security;

/// <summary>
/// Resolves the tier from the authenticated request principal: the API key's tier claim, or for a JWT caller the
/// claim named by <see cref="ServingJwtTierOptions.TierClaimType"/>.
/// </summary>
public sealed class ClaimsTierResolver : ITierResolver
{
    private readonly TierEnforcementOptions _options;
    private readonly string _jwtTierClaimType;

    public ClaimsTierResolver(IOptions<TierEnforcementOptions> options, IOptions<ServingSecurityOptions> security)
    {
        Guard.NotNull(options);
        _options = options.Value;
        Guard.NotNull(security);
        _jwtTierClaimType = security.Value.Jwt.TierClaimType;
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

        var tierClaim = user.FindFirst(ApiKeyClaimTypes.Tier)?.Value ?? user.FindFirst(_jwtTierClaimType)?.Value;
        if (string.IsNullOrWhiteSpace(tierClaim))
        {
            return _options.DefaultTier;
        }

        // Names only: Enum.TryParse also accepts undefined numbers such as "999".
        return Enum.TryParse<SubscriptionTier>(tierClaim, ignoreCase: true, out var tier) && Enum.IsDefined(tier)
            ? tier
            : _options.DefaultTier;
    }
}

