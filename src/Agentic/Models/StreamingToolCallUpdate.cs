namespace AiDotNet.Agentic.Models;

/// <summary>
/// An incremental fragment of a tool call arriving over a streaming response.
/// </summary>
/// <remarks>
/// <para>
/// When a model streams a tool call, the pieces arrive across several chunks: the id and tool name
/// usually come first, then the JSON arguments stream in fragments that must be concatenated by
/// <see cref="Index"/>. Accumulating these fragments yields a complete <see cref="ToolCallContent"/>.
/// </para>
/// <para><b>For Beginners:</b> While streaming, a tool call is delivered a little at a time — first
/// "I'm going to call tool #0 named <c>get_weather</c>", then the arguments dribble in like
/// <c>{"ci</c>, <c>ty":"Par</c>, <c>is"}</c>. The <see cref="Index"/> tells you which tool call a
/// fragment belongs to so you can stitch the right pieces together.
/// </para>
/// </remarks>
public sealed class StreamingToolCallUpdate
{
    /// <summary>
    /// Initializes a new streaming tool-call fragment.
    /// </summary>
    /// <param name="index">Zero-based position of this tool call within the response's tool-call list.</param>
    /// <param name="callId">The tool-call id, when present in this fragment.</param>
    /// <param name="toolName">The tool name, when present in this fragment.</param>
    /// <param name="argumentsJsonFragment">A fragment of the arguments JSON to append, when present.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative.</exception>
    public StreamingToolCallUpdate(
        int index,
        string? callId = null,
        string? toolName = null,
        string? argumentsJsonFragment = null)
    {
        Guard.NonNegative(index);
        Index = index;
        CallId = callId;
        ToolName = toolName;
        ArgumentsJsonFragment = argumentsJsonFragment;
    }

    /// <summary>
    /// Gets the zero-based index identifying which tool call this fragment contributes to.
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the tool-call id, or <c>null</c> when this fragment does not carry it.
    /// </summary>
    public string? CallId { get; }

    /// <summary>
    /// Gets the tool name, or <c>null</c> when this fragment does not carry it.
    /// </summary>
    public string? ToolName { get; }

    /// <summary>
    /// Gets a fragment of the arguments JSON to append, or <c>null</c> when this fragment carries none.
    /// </summary>
    public string? ArgumentsJsonFragment { get; }
}
