using AiDotNet.Serving.Configuration;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Security;

/// <summary>
/// Resolves the tier from a configurable HTTP header.
/// </summary>
public sealed class HeaderTierResolver : ITierResolver
{
    private readonly TierEnforcementOptions _options;

    public HeaderTierResolver(IOptions<TierEnforcementOptions> options)
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

        if (!httpContext.Request.Headers.TryGetValue(_options.TierHeaderName, out var values))
        {
            return _options.DefaultTier;
        }

        var raw = values.ToString();
        if (string.IsNullOrWhiteSpace(raw))
        {
            return _options.DefaultTier;
        }

        if (Enum.TryParse<SubscriptionTier>(raw.Trim(), ignoreCase: true, out var tier))
        {
            return tier;
        }

        return _options.DefaultTier;
    }
}

