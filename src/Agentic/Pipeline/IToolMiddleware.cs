using AiDotNet.Agentic.Tools;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// The delegate that invokes the next stage of a tool-invocation pipeline (ultimately the tool itself).
/// </summary>
/// <param name="context">The (possibly rewritten) invocation context.</param>
/// <param name="cancellationToken">Token used to cancel the call.</param>
/// <returns>The tool's (or downstream middleware's) result.</returns>
public delegate Task<ToolInvocationResult> ToolPipelineDelegate(ToolInvocationContext context, CancellationToken cancellationToken);

/// <summary>
/// A cross-cutting filter around tool/function execution (the AiDotNet analogue of Semantic Kernel's
/// function-invocation filter). Middleware can rewrite arguments, require approval, log, time, cache, or
/// short-circuit a tool call (e.g., deny it) by returning a result without calling <paramref name="next"/>.
/// </summary>
/// <remarks>
/// <para>
/// Applied by wrapping a tool in a <see cref="MiddlewareAgentTool"/>; the wrapped tool is registered like any
/// other, so the agent loop runs the middleware transparently whenever the model calls it. This is the hook
/// for human-in-the-loop tool approval and tool-level guardrails/observability.
/// </para>
/// <para><b>For Beginners:</b> A reusable wrapper around running a tool — ask for confirmation before a risky
/// action, log every call, fix up the inputs, or block it outright — without changing the tool's code.
/// </para>
/// </remarks>
public interface IToolMiddleware
{
    /// <summary>
    /// Processes a tool call, optionally calling <paramref name="next"/> to run the tool.
    /// </summary>
    /// <param name="context">The invocation context (mutable).</param>
    /// <param name="next">Runs the next stage / the tool; skip it to short-circuit.</param>
    /// <param name="cancellationToken">Token used to cancel the call.</param>
    /// <returns>The result to return upstream.</returns>
    Task<ToolInvocationResult> InvokeAsync(
        ToolInvocationContext context,
        ToolPipelineDelegate next,
        CancellationToken cancellationToken);
}
