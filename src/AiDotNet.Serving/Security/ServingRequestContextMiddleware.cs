using System.Text.Json;

namespace AiDotNet.Serving.Security;

public sealed class ServingRequestContextMiddleware
{
    private readonly RequestDelegate _next;

    public ServingRequestContextMiddleware(RequestDelegate next)
    {
        _next = next ?? throw new ArgumentNullException(nameof(next));
    }

    public async Task InvokeAsync(
        HttpContext context,
        IServingRequestContextResolver resolver,
        IServingRequestContextAccessor accessor)
    {
        try
        {
            accessor.Current = await resolver.ResolveAsync(context, context.RequestAborted).ConfigureAwait(false);
        }
        catch (UnauthorizedAccessException ex)
        {
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            context.Response.ContentType = "application/json";
            await context.Response.WriteAsync(
                JsonSerializer.Serialize(new { error = ex.Message }),
                context.RequestAborted).ConfigureAwait(false);
            return;
        }

        await _next(context).ConfigureAwait(false);
    }
}

