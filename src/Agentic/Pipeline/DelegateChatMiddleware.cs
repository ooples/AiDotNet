using AiDotNet.Agentic.Models;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IChatMiddleware"/> backed by a delegate — the quick way to add a one-off filter (logging,
/// a header injection, a guard) without declaring a class.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The do-it-yourself middleware: hand it a small function that receives the
/// request and the "call the rest of the pipeline" handle, and it becomes a reusable filter.
/// </para>
/// </remarks>
public sealed class DelegateChatMiddleware : IChatMiddleware
{
    private readonly Func<ChatRequestContext, ChatPipelineDelegate, CancellationToken, Task<ChatResponse>> _handler;

    /// <summary>
    /// Initializes a new delegate middleware.
    /// </summary>
    /// <param name="handler">The function implementing the middleware behavior.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="handler"/> is <c>null</c>.</exception>
    public DelegateChatMiddleware(Func<ChatRequestContext, ChatPipelineDelegate, CancellationToken, Task<ChatResponse>> handler)
    {
        Guard.NotNull(handler);
        _handler = handler;
    }

    /// <inheritdoc/>
    public Task<ChatResponse> InvokeAsync(ChatRequestContext context, ChatPipelineDelegate next, CancellationToken cancellationToken)
    {
        Guard.NotNull(context);
        Guard.NotNull(next);
        return _handler(context, next, cancellationToken);
    }
}
