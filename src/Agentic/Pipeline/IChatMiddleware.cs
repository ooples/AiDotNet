using AiDotNet.Agentic.Models;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// The delegate that invokes the next stage of a chat-middleware pipeline (ultimately the model call).
/// </summary>
/// <param name="context">The (possibly rewritten) request context.</param>
/// <param name="cancellationToken">Token used to cancel the call.</param>
/// <returns>The downstream response.</returns>
public delegate Task<ChatResponse> ChatPipelineDelegate(ChatRequestContext context, CancellationToken cancellationToken);

/// <summary>
/// A cross-cutting filter around chat-model calls (the AiDotNet analogue of Semantic Kernel filters / a
/// middleware pipeline). Each middleware wraps the next stage: it can modify the request before calling
/// <paramref name="next"/>, inspect or replace the response after, short-circuit by returning without calling
/// next (caching, guardrails, mocking), retry, log, or measure.
/// </summary>
/// <remarks>
/// <para>
/// Middleware are composed by <see cref="MiddlewareChatClient{T}"/> in registration order (the first
/// registered runs outermost). Because the request/response types are not numeric-generic, middleware are
/// written once and apply to any <see cref="IChatClient{T}"/> backend — cloud or local.
/// </para>
/// <para><b>For Beginners:</b> A reusable wrapper you can put around <em>every</em> model call: log it, time
/// it, add a standing instruction, block unsafe content, cache repeats, or retry on failure — without
/// touching the agent or the model. Stack several and they run in order.
/// </para>
/// </remarks>
public interface IChatMiddleware
{
    /// <summary>
    /// Processes a chat call, optionally calling <paramref name="next"/> to continue down the pipeline.
    /// </summary>
    /// <param name="context">The request context (mutable).</param>
    /// <param name="next">Invokes the next stage; skip it to short-circuit.</param>
    /// <param name="cancellationToken">Token used to cancel the call.</param>
    /// <returns>The response to return upstream.</returns>
    Task<ChatResponse> InvokeAsync(
        ChatRequestContext context,
        ChatPipelineDelegate next,
        CancellationToken cancellationToken);
}
