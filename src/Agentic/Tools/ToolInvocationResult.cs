namespace AiDotNet.Agentic.Tools;

/// <summary>
/// The outcome of executing an <see cref="IAgentTool"/>: the text fed back to the model plus a flag
/// indicating whether the tool failed.
/// </summary>
/// <remarks>
/// <para>
/// Tools return results rather than throwing for expected failures, so the model can see the error
/// and recover (for example, retry with different arguments). The <see cref="Content"/> is what gets
/// placed into the <see cref="AiDotNet.Agentic.Models.ToolResultContent"/> sent back to the model.
/// </para>
/// <para><b>For Beginners:</b> When a tool runs, it produces an answer string. This wraps that answer
/// together with a yes/no flag for "did it go wrong?". Use <see cref="Success"/> for a good result and
/// <see cref="Error"/> for a failure message.
/// </para>
/// </remarks>
public sealed class ToolInvocationResult
{
    private ToolInvocationResult(string content, bool isError)
    {
        Content = content;
        IsError = isError;
    }

    /// <summary>
    /// Gets the result text to feed back to the model (plain text or serialized JSON).
    /// </summary>
    public string Content { get; }

    /// <summary>
    /// Gets a value indicating whether the tool invocation failed.
    /// </summary>
    public bool IsError { get; }

    /// <summary>
    /// Creates a successful result.
    /// </summary>
    /// <param name="content">The tool's output. May be empty, not <c>null</c>.</param>
    /// <returns>A successful <see cref="ToolInvocationResult"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="content"/> is <c>null</c>.</exception>
    public static ToolInvocationResult Success(string content)
    {
        Guard.NotNull(content);
        return new ToolInvocationResult(content, isError: false);
    }

    /// <summary>
    /// Creates a failed result carrying an error message.
    /// </summary>
    /// <param name="message">A human/model-readable description of what went wrong.</param>
    /// <returns>A failed <see cref="ToolInvocationResult"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="message"/> is <c>null</c>.</exception>
    public static ToolInvocationResult Error(string message)
    {
        Guard.NotNull(message);
        return new ToolInvocationResult(message, isError: true);
    }
}
