namespace AiDotNet.Serving.Security;

public interface IServingRequestContextResolver
{
    Task<ServingRequestContext> ResolveAsync(HttpContext httpContext, CancellationToken cancellationToken);
}

