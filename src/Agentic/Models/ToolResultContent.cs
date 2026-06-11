namespace AiDotNet.Agentic.Models;

/// <summary>
/// The result of executing a tool, fed back to the model to continue a tool-calling turn.
/// </summary>
/// <remarks>
/// <para>
/// After the assistant emits a <see cref="ToolCallContent"/> and the caller runs the tool, the output
/// is returned to the model as a <see cref="ChatRole.Tool"/> message containing this content. The
/// <see cref="CallId"/> must match the originating call so the model knows which request this answers.
/// </para>
/// <para><b>For Beginners:</b> This is the answer slip you hand back after running the tool the model
/// asked for. It carries the same ticket number (<see cref="CallId"/>) as the request, the tool's
/// output (<see cref="Result"/>), and a flag (<see cref="IsError"/>) so the model knows whether the
/// tool succeeded or failed.
/// </para>
/// </remarks>
public sealed class ToolResultContent : AiContent
{
    /// <summary>
    /// Initializes a new tool result.
    /// </summary>
    /// <param name="callId">The id of the <see cref="ToolCallContent"/> this result answers.</param>
    /// <param name="result">The tool output (text or serialized JSON). May be empty, not <c>null</c>.</param>
    /// <param name="isError">Whether the tool failed; lets the model react to the failure.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="callId"/> or <paramref name="result"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="callId"/> is empty/whitespace.</exception>
    public ToolResultContent(string callId, string result, bool isError = false)
    {
        Guard.NotNullOrWhiteSpace(callId);
        Guard.NotNull(result);
        CallId = callId;
        Result = result;
        IsError = isError;
    }

    /// <summary>
    /// Gets the id of the tool call this result corresponds to.
    /// </summary>
    public string CallId { get; }

    /// <summary>
    /// Gets the tool's output as a string (plain text or serialized JSON).
    /// </summary>
    public string Result { get; }

    /// <summary>
    /// Gets a value indicating whether the tool invocation failed.
    /// </summary>
    public bool IsError { get; }

    /// <inheritdoc/>
    public override string ToString() => IsError ? $"[error:{CallId}] {Result}" : $"[{CallId}] {Result}";
}
