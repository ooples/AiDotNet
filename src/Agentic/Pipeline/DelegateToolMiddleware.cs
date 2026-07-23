using AiDotNet.Agentic.Tools;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IToolMiddleware"/> backed by a delegate — the quick way to add a one-off tool filter
/// (logging, argument fix-up, a guard) without declaring a class.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The do-it-yourself tool filter: give it a function that receives the call and
/// the "run the tool" handle, and it becomes a reusable wrapper.
/// </para>
/// </remarks>
public sealed class DelegateToolMiddleware : IToolMiddleware
{
    private readonly Func<ToolInvocationContext, ToolPipelineDelegate, CancellationToken, Task<ToolInvocationResult>> _handler;

    /// <summary>
    /// Initializes a new delegate tool middleware.
    /// </summary>
    /// <param name="handler">The function implementing the middleware behavior.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="handler"/> is <c>null</c>.</exception>
    public DelegateToolMiddleware(Func<ToolInvocationContext, ToolPipelineDelegate, CancellationToken, Task<ToolInvocationResult>> handler)
    {
        Guard.NotNull(handler);
        _handler = handler;
    }

    /// <inheritdoc/>
    public Task<ToolInvocationResult> InvokeAsync(ToolInvocationContext context, ToolPipelineDelegate next, CancellationToken cancellationToken)
    {
        Guard.NotNull(context);
        Guard.NotNull(next);
        return _handler(context, next, cancellationToken);
    }
}
