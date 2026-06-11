namespace AiDotNet.Agentic.Models;

/// <summary>
/// Describes why a chat model stopped generating a response.
/// </summary>
/// <remarks>
/// <para>
/// Every completed (or streamed) response carries a finish reason so callers can react correctly:
/// for example, retrying with a larger token budget on <see cref="Length"/>, or executing the
/// requested tools on <see cref="ToolCalls"/>.
/// </para>
/// <para><b>For Beginners:</b> When the model stops "typing", it tells you why. Did it finish its
/// thought naturally (<see cref="Stop"/>)? Did it run out of room (<see cref="Length"/>)? Did it
/// pause to ask for a tool to be run (<see cref="ToolCalls"/>)? Was the output blocked by a safety
/// filter (<see cref="ContentFilter"/>)? This enum captures those outcomes so your code can decide
/// what to do next.
/// </para>
/// </remarks>
public enum ChatFinishReason
{
    /// <summary>
    /// The model finished naturally (it reached a stopping point or a configured stop sequence).
    /// </summary>
    Stop,

    /// <summary>
    /// Generation was cut off because the maximum output-token budget was reached. The response is
    /// likely incomplete; consider increasing <c>MaxOutputTokens</c> and retrying.
    /// </summary>
    Length,

    /// <summary>
    /// The model paused to request one or more tool/function calls. The caller is expected to execute
    /// the tools and feed the results back as <see cref="ChatRole.Tool"/> messages, then continue.
    /// </summary>
    ToolCalls,

    /// <summary>
    /// Output was withheld or truncated by a content-safety filter.
    /// </summary>
    ContentFilter,

    /// <summary>
    /// The provider returned a finish reason that does not map to any of the known values.
    /// </summary>
    Unknown
}
