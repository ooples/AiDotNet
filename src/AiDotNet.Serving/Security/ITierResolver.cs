using Microsoft.AspNetCore.Http;

namespace AiDotNet.Serving.Security;

/// <summary>
/// Resolves the subscription tier for an incoming HTTP request.
/// </summary>
public interface ITierResolver
{
    /// <summary>
    /// Resolves the tier from the request context.
    /// </summary>
    SubscriptionTier ResolveTier(HttpContext httpContext);
}

