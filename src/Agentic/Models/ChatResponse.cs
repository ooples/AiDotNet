namespace AiDotNet.Agentic.Models;

/// <summary>
/// The complete result of a non-streaming chat call.
/// </summary>
/// <remarks>
/// <para>
/// Wraps the assistant <see cref="Message"/> the model produced (which may contain text and/or
/// <see cref="ToolCallContent"/> parts), why generation stopped (<see cref="FinishReason"/>), token
/// <see cref="Usage"/>, and the model id that served the request.
/// </para>
/// <para><b>For Beginners:</b> This is everything you get back from one call: the reply itself, the
/// reason it stopped, and how many tokens it cost. If <see cref="FinishReason"/> is
/// <see cref="ChatFinishReason.ToolCalls"/>, the reply is asking you to run tools rather than giving a
/// final answer.
/// </para>
/// </remarks>
public sealed class ChatResponse
{
    /// <summary>
    /// Initializes a new <see cref="ChatResponse"/>.
    /// </summary>
    /// <param name="message">The assistant message produced by the model.</param>
    /// <param name="finishReason">Why generation stopped.</param>
    /// <param name="usage">Token usage, when reported by the provider.</param>
    /// <param name="modelId">The id of the model that served the request, when reported.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="message"/> is <c>null</c>.</exception>
    public ChatResponse(
        ChatMessage message,
        ChatFinishReason finishReason = ChatFinishReason.Stop,
        ChatUsage? usage = null,
        string? modelId = null)
    {
        Guard.NotNull(message);
        Message = message;
        FinishReason = finishReason;
        Usage = usage;
        ModelId = modelId;
    }

    /// <summary>
    /// Gets the assistant message produced by the model.
    /// </summary>
    public ChatMessage Message { get; }

    /// <summary>
    /// Gets the reason generation stopped.
    /// </summary>
    public ChatFinishReason FinishReason { get; }

    /// <summary>
    /// Gets token usage for the request, or <c>null</c> when the provider did not report it.
    /// </summary>
    public ChatUsage? Usage { get; }

    /// <summary>
    /// Gets the id of the model that served the request, or <c>null</c> when not reported.
    /// </summary>
    public string? ModelId { get; }

    /// <summary>
    /// Gets the concatenated text of the assistant message (shortcut for <c>Message.Text</c>).
    /// </summary>
    public string Text => Message.Text;
}
