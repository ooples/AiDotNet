namespace AiDotNet.Agentic.Models;

/// <summary>
/// A request, emitted by the assistant, to invoke a named tool/function with JSON arguments.
/// </summary>
/// <remarks>
/// <para>
/// This is the heart of native function calling. Instead of the model writing "please run the
/// calculator" as prose for us to parse, the provider returns a structured tool-call: a stable
/// <see cref="CallId"/>, the <see cref="ToolName"/> to invoke, and the <see cref="ArgumentsJson"/>
/// the model chose. The caller executes the tool and replies with a matching
/// <see cref="ToolResultContent"/> carrying the same <see cref="CallId"/>.
/// </para>
/// <para><b>For Beginners:</b> When the model wants to use a tool, it doesn't run the tool itself —
/// it hands you a filled-in request form: "call the tool named <c>get_weather</c> with
/// <c>{\"city\":\"Paris\"}</c>, and here's a ticket number so we can match up the answer." This class
/// is that form. The ticket number is <see cref="CallId"/>.
/// </para>
/// </remarks>
public sealed class ToolCallContent : AiContent
{
    /// <summary>
    /// Initializes a new tool-call request.
    /// </summary>
    /// <param name="callId">Provider-assigned identifier used to correlate the eventual result.</param>
    /// <param name="toolName">The name of the tool the model wants to invoke.</param>
    /// <param name="argumentsJson">The arguments as a raw JSON object string. Defaults to <c>{}</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="callId"/> or <paramref name="toolName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="callId"/> or <paramref name="toolName"/> is empty/whitespace.</exception>
    /// <remarks>
    /// <paramref name="argumentsJson"/> is intentionally NOT validated here. The model produces these
    /// arguments, and malformed JSON is an expected runtime condition: <c>ToolCollection.InvokeAsync</c>
    /// detects it and returns an error <see cref="ToolResultContent"/> so the agent can feed the failure back
    /// to the model for correction. Throwing at construction would instead crash the whole agent loop on a
    /// single bad tool call. Null/whitespace defaults to <c>{}</c>.
    /// </remarks>
    public ToolCallContent(string callId, string toolName, string? argumentsJson = null)
    {
        Guard.NotNullOrWhiteSpace(callId);
        Guard.NotNullOrWhiteSpace(toolName);
        CallId = callId;
        ToolName = toolName;
        ArgumentsJson = argumentsJson is null || argumentsJson.Trim().Length == 0 ? "{}" : argumentsJson;
    }

    /// <summary>
    /// Gets the provider-assigned id correlating this call to its <see cref="ToolResultContent"/>.
    /// </summary>
    public string CallId { get; }

    /// <summary>
    /// Gets the name of the tool the model requested.
    /// </summary>
    public string ToolName { get; }

    /// <summary>
    /// Gets the model-chosen arguments as a raw JSON object string (never <c>null</c>; defaults to <c>{}</c>).
    /// </summary>
    public string ArgumentsJson { get; }

    /// <inheritdoc/>
    public override string ToString() => $"{ToolName}({ArgumentsJson}) [{CallId}]";
}
