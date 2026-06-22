namespace AiDotNet.Agentic.Models;

/// <summary>
/// A single incremental update in a streaming chat response.
/// </summary>
/// <remarks>
/// <para>
/// Streaming delivers a response as a sequence of these updates instead of one final object. A typical
/// stream carries the role once, then many <see cref="TextDelta"/> chunks (and/or
/// <see cref="ToolCall"/> fragments), and finally a <see cref="FinishReason"/> with optional
/// <see cref="Usage"/>. Concatenating the deltas reconstructs the full reply.
/// </para>
/// <para><b>For Beginners:</b> This is one "frame" of the model typing in real time. Each frame usually
/// holds the next little bit of text (<see cref="TextDelta"/>). The last frame tells you it finished
/// and how many tokens were used. Use the factory methods (<see cref="ForText"/>, <see cref="ForToolCall"/>,
/// <see cref="ForFinish"/>) to build updates without juggling many constructor arguments.
/// </para>
/// </remarks>
public sealed class ChatResponseUpdate
{
    /// <summary>
    /// Initializes a new streaming update. All parts are optional; a given update typically carries only one.
    /// </summary>
    /// <param name="role">The author role, usually present on the first update.</param>
    /// <param name="textDelta">The next fragment of assistant text, if any.</param>
    /// <param name="toolCall">A streaming tool-call fragment, if any.</param>
    /// <param name="finishReason">The finish reason, present on the final update.</param>
    /// <param name="usage">Token usage, present on the final update for providers that report it.</param>
    public ChatResponseUpdate(
        ChatRole? role = null,
        string? textDelta = null,
        StreamingToolCallUpdate? toolCall = null,
        ChatFinishReason? finishReason = null,
        ChatUsage? usage = null)
    {
        Role = role;
        TextDelta = textDelta;
        ToolCall = toolCall;
        FinishReason = finishReason;
        Usage = usage;
    }

    /// <summary>
    /// Gets the author role for the response, when carried by this update.
    /// </summary>
    public ChatRole? Role { get; }

    /// <summary>
    /// Gets the next fragment of assistant text, or <c>null</c> when this update carries none.
    /// </summary>
    public string? TextDelta { get; }

    /// <summary>
    /// Gets a streaming tool-call fragment, or <c>null</c> when this update carries none.
    /// </summary>
    public StreamingToolCallUpdate? ToolCall { get; }

    /// <summary>
    /// Gets the finish reason, present only on the terminal update.
    /// </summary>
    public ChatFinishReason? FinishReason { get; }

    /// <summary>
    /// Gets token usage, present only on the terminal update for providers that report it mid-stream.
    /// </summary>
    public ChatUsage? Usage { get; }

    /// <summary>
    /// Creates an update carrying a fragment of assistant text.
    /// </summary>
    /// <param name="textDelta">The text fragment.</param>
    /// <returns>A new text update.</returns>
    public static ChatResponseUpdate ForText(string textDelta) => new(textDelta: textDelta);

    /// <summary>
    /// Creates an update carrying a streaming tool-call fragment.
    /// </summary>
    /// <param name="toolCall">The tool-call fragment.</param>
    /// <returns>A new tool-call update.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="toolCall"/> is <c>null</c>.</exception>
    public static ChatResponseUpdate ForToolCall(StreamingToolCallUpdate toolCall)
    {
        Guard.NotNull(toolCall);
        return new ChatResponseUpdate(toolCall: toolCall);
    }

    /// <summary>
    /// Creates the terminal update carrying the finish reason and optional usage.
    /// </summary>
    /// <param name="finishReason">Why generation stopped.</param>
    /// <param name="usage">Token usage, if known.</param>
    /// <returns>A new terminal update.</returns>
    public static ChatResponseUpdate ForFinish(ChatFinishReason finishReason, ChatUsage? usage = null) =>
        new(finishReason: finishReason, usage: usage);
}
