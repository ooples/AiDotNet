using System.Security.Claims;

namespace AiDotNet.Serving.Security;

public sealed class ServingRequestContextResolver : IServingRequestContextResolver
{
    private readonly ITierResolver _tierResolver;

    public ServingRequestContextResolver(
        ITierResolver tierResolver)
    {
        _tierResolver = tierResolver ?? throw new ArgumentNullException(nameof(tierResolver));
    }

    public Task<ServingRequestContext> ResolveAsync(HttpContext httpContext, CancellationToken cancellationToken)
    {
        _ = cancellationToken;

        if (httpContext is null)
        {
            throw new ArgumentNullException(nameof(httpContext));
        }

        var isAuthenticated = httpContext.User?.Identity?.IsAuthenticated == true;
        var subscriptionTier = _tierResolver.ResolveTier(httpContext);

        return Task.FromResult(new ServingRequestContext
        {
            Tier = MapTier(subscriptionTier),
            IsAuthenticated = isAuthenticated,
            Subject = isAuthenticated ? ResolveSubject(httpContext.User) : null
        });
    }

    private static ServingTier MapTier(SubscriptionTier tier) =>
        tier switch
        {
            SubscriptionTier.Pro => ServingTier.Premium,
            SubscriptionTier.Enterprise => ServingTier.Enterprise,
            _ => ServingTier.Free
        };

    private static string? ResolveSubject(ClaimsPrincipal principal) =>
        principal.FindFirstValue("sub") ??
        principal.FindFirstValue(ClaimTypes.NameIdentifier) ??
        principal.Identity?.Name;
}

