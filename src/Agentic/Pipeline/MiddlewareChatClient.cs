using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IChatClient{T}"/> decorator that runs a chain of <see cref="IChatMiddleware"/> around an
/// inner client's calls — the composition root for filters/middleware (logging, guardrails, caching, retry,
/// telemetry). The first-registered middleware runs outermost.
/// </summary>
/// <typeparam name="T">The numeric type of the wrapped client.</typeparam>
/// <remarks>
/// <para>
/// Middleware apply to <see cref="GetResponseAsync"/> (the complete-response path), where pre/post processing
/// and short-circuiting are well-defined. <see cref="GetStreamingResponseAsync"/> passes through to the inner
/// client unchanged, since wrapping a token stream has different semantics; stream-aware middleware is a
/// separate concern.
/// </para>
/// <para><b>For Beginners:</b> Wrap your model client in this with a list of filters, and every
/// (non-streaming) call flows through them in order before and after hitting the model — one place to add
/// behavior that should apply everywhere.
/// </para>
/// </remarks>
public sealed class MiddlewareChatClient<T> : IChatClient<T>
{
    private readonly IChatClient<T> _inner;
    private readonly IReadOnlyList<IChatMiddleware> _middlewares;

    /// <summary>
    /// Initializes a new middleware client.
    /// </summary>
    /// <param name="inner">The client the pipeline ultimately calls.</param>
    /// <param name="middlewares">The middleware to run, outermost first. May be empty.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="middlewares"/> (or any element) is <c>null</c>.</exception>
    public MiddlewareChatClient(IChatClient<T> inner, IReadOnlyList<IChatMiddleware> middlewares)
    {
        Guard.NotNull(inner);
        Guard.NotNull(middlewares);
        var copy = new List<IChatMiddleware>(middlewares.Count);
        foreach (var middleware in middlewares)
        {
            Guard.NotNull(middleware);
            copy.Add(middleware);
        }

        _inner = inner;
        _middlewares = copy;
    }

    /// <inheritdoc/>
    public string ModelId => _inner.ModelId;

    /// <inheritdoc/>
    public Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);

        var context = new ChatRequestContext(messages, options);

        ChatPipelineDelegate pipeline = (ctx, ct) =>
            _inner.GetResponseAsync(ctx.Messages, ctx.Options, ct);

        for (var i = _middlewares.Count - 1; i >= 0; i--)
        {
            var middleware = _middlewares[i];
            var next = pipeline;
            pipeline = (ctx, ct) => middleware.InvokeAsync(ctx, next, ct);
        }

        return pipeline(context, cancellationToken);
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        _inner.GetStreamingResponseAsync(messages, options, cancellationToken);
}
