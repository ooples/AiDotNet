using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IAgentTool"/> decorator that runs a chain of <see cref="IToolMiddleware"/> around an inner
/// tool's execution. Register the wrapped tool like any other and the agent loop applies the middleware
/// transparently whenever the model calls it — the composition root for tool approval, guardrails, logging,
/// and caching.
/// </summary>
/// <remarks>
/// <para>
/// The tool's identity (name, description, schema, definition) is delegated to the inner tool, so wrapping is
/// invisible to the model — only execution is intercepted. Middleware run in registration order (first =
/// outermost).
/// </para>
/// <para><b>For Beginners:</b> Wrap a tool with this and a list of filters, and every time the model uses that
/// tool the filters run first (approve it, log it, fix the inputs) — the model still sees the same tool.
/// </para>
/// </remarks>
public sealed class MiddlewareAgentTool : IAgentTool
{
    private readonly IAgentTool _inner;
    private readonly IReadOnlyList<IToolMiddleware> _middlewares;

    /// <summary>
    /// Initializes a new middleware tool decorator.
    /// </summary>
    /// <param name="inner">The tool the pipeline ultimately runs.</param>
    /// <param name="middlewares">The middleware to apply, outermost first. May be empty.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="middlewares"/> (or any element) is <c>null</c>.</exception>
    public MiddlewareAgentTool(IAgentTool inner, IReadOnlyList<IToolMiddleware> middlewares)
    {
        Guard.NotNull(inner);
        Guard.NotNull(middlewares);
        var copy = new List<IToolMiddleware>(middlewares.Count);
        foreach (var middleware in middlewares)
        {
            Guard.NotNull(middleware);
            copy.Add(middleware);
        }

        _inner = inner;
        _middlewares = copy;
    }

    /// <inheritdoc/>
    public string Name => _inner.Name;

    /// <inheritdoc/>
    public string Description => _inner.Description;

    /// <inheritdoc/>
    public JObject ParametersSchema => _inner.ParametersSchema;

    /// <inheritdoc/>
    public AiToolDefinition ToDefinition() => _inner.ToDefinition();

    /// <inheritdoc/>
    public Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(arguments);

        // Deep-clone the arguments: middleware may rewrite ctx.Arguments, and
        // those rewrites must stay inside the pipeline rather than mutating
        // the caller's original payload.
        var context = new ToolInvocationContext(_inner.Name, (JObject)arguments.DeepClone());

        ToolPipelineDelegate pipeline = (ctx, ct) => _inner.InvokeAsync(ctx.Arguments, ct);
        for (var i = _middlewares.Count - 1; i >= 0; i--)
        {
            var middleware = _middlewares[i];
            var next = pipeline;
            pipeline = (ctx, ct) => middleware.InvokeAsync(ctx, next, ct);
        }

        return pipeline(context, cancellationToken);
    }
}
