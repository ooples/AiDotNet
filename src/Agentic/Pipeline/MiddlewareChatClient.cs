using System.Runtime.CompilerServices;
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
/// Middleware semantics (pre/post processing, short-circuiting) are defined on the complete-response path.
/// When middleware are configured, <see cref="GetStreamingResponseAsync"/> therefore runs the full pipeline
/// and re-emits the final response as a stream — streaming callers get the exact same policy enforcement as
/// non-streaming callers, at the cost of token-level incrementality. With no middleware configured, streaming
/// passes through to the inner client natively.
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
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);

        if (_middlewares.Count == 0)
        {
            await foreach (var update in _inner.GetStreamingResponseAsync(messages, options, cancellationToken)
                .ConfigureAwait(false))
            {
                yield return update;
            }

            yield break;
        }

        // Guardrails/logging/retry configured on this client MUST apply to
        // streaming calls too — silently bypassing them would let the same
        // wrapped client enforce policy on one path but not the other. Run the
        // complete-response pipeline and re-emit the result as a stream.
        var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        yield return new ChatResponseUpdate(role: ChatRole.Assistant);
        if (response.Message.Text.Length > 0)
        {
            yield return ChatResponseUpdate.ForText(response.Message.Text);
        }

        var toolCalls = response.Message.ToolCalls;
        for (var i = 0; i < toolCalls.Count; i++)
        {
            var call = toolCalls[i];
            yield return ChatResponseUpdate.ForToolCall(
                new StreamingToolCallUpdate(i, call.CallId, call.ToolName, call.ArgumentsJson));
        }

        yield return ChatResponseUpdate.ForFinish(response.FinishReason, response.Usage);
    }
}
