using AiDotNet.Agentic.Tools;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IToolMiddleware"/> that gates tool execution behind an approval check — human-in-the-loop or
/// policy-based tool authorization. When the check denies a call, the tool does not run and a deny result is
/// returned to the model so it can react.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A permission gate for tools. Before a (say) "delete file" or "send email" tool
/// runs, your approval function decides yes/no. On "no", the tool is skipped and the model is told it wasn't
/// allowed. Wrap sensitive tools with this to keep a human (or a rule) in control.
/// </para>
/// </remarks>
public sealed class ApprovalToolMiddleware : IToolMiddleware
{
    private readonly Func<ToolInvocationContext, bool> _approve;
    private readonly string? _denyMessage;

    /// <summary>
    /// Initializes a new approval middleware.
    /// </summary>
    /// <param name="approve">Returns <c>true</c> to allow the call, <c>false</c> to deny it.</param>
    /// <param name="denyMessage">The message returned when denied. <c>null</c> uses a default mentioning the tool.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="approve"/> is <c>null</c>.</exception>
    public ApprovalToolMiddleware(Func<ToolInvocationContext, bool> approve, string? denyMessage = null)
    {
        Guard.NotNull(approve);
        _approve = approve;
        _denyMessage = denyMessage;
    }

    /// <inheritdoc/>
    public Task<ToolInvocationResult> InvokeAsync(ToolInvocationContext context, ToolPipelineDelegate next, CancellationToken cancellationToken)
    {
        Guard.NotNull(context);
        Guard.NotNull(next);

        if (!_approve(context))
        {
            var message = _denyMessage is { } m && m.Trim().Length > 0
                ? m
                : $"Tool call '{context.ToolName}' was denied and not executed.";
            return Task.FromResult(ToolInvocationResult.Error(message));
        }

        return next(context, cancellationToken);
    }
}
