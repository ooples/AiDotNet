namespace AiDotNet.Agentic.Models;

/// <summary>
/// A single message in a chat conversation: a <see cref="ChatRole"/> plus one or more content parts.
/// </summary>
/// <remarks>
/// <para>
/// A chat request is an ordered list of <see cref="ChatMessage"/> values. Each message is authored by
/// a role (system/user/assistant/tool) and carries a list of <see cref="AiContent"/> parts so it can
/// mix text, images, tool-call requests, and tool results. Messages are immutable once constructed.
/// </para>
/// <para><b>For Beginners:</b> This is one line in the conversation transcript. The most common case is
/// a bit of text from the user or the assistant, so there are shortcuts for that:
/// <c>ChatMessage.User("Hello")</c>, <c>ChatMessage.System("You are helpful")</c>,
/// <c>ChatMessage.Assistant("Hi!")</c>. For tool calling there are richer parts, but the shortcuts
/// cover everyday use.
/// </para>
/// </remarks>
public sealed class ChatMessage
{
    /// <summary>
    /// Initializes a new message from a role and an explicit list of content parts.
    /// </summary>
    /// <param name="role">The author role.</param>
    /// <param name="contents">The content parts. Must be non-null and contain no null elements.</param>
    /// <param name="authorName">Optional author name (e.g., a specific tool or participant name).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="contents"/> or any element is <c>null</c>.</exception>
    public ChatMessage(ChatRole role, IReadOnlyList<AiContent> contents, string? authorName = null)
    {
        Guard.NotNull(contents);
        if (contents.Count == 0)
        {
            // The type contract is "one or more content parts" — reject the
            // empty list at the boundary so downstream connectors don't see
            // invalid messages.
            throw new ArgumentException(
                "At least one content part is required.", nameof(contents));
        }

        var copy = new List<AiContent>(contents.Count);
        foreach (var part in contents)
        {
            Guard.NotNull(part);
            copy.Add(part);
        }

        Role = role;
        Contents = copy;
        AuthorName = authorName;
    }

    /// <summary>
    /// Initializes a new message from a role and a single piece of text.
    /// </summary>
    /// <param name="role">The author role.</param>
    /// <param name="text">The message text.</param>
    /// <param name="authorName">Optional author name.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is <c>null</c>.</exception>
    public ChatMessage(ChatRole role, string text, string? authorName = null)
        : this(role, new AiContent[] { new TextContent(text) }, authorName)
    {
    }

    /// <summary>
    /// Gets the role that authored this message.
    /// </summary>
    public ChatRole Role { get; }

    /// <summary>
    /// Gets the ordered, immutable list of content parts that make up this message.
    /// </summary>
    public IReadOnlyList<AiContent> Contents { get; }

    /// <summary>
    /// Gets the optional author name associated with this message.
    /// </summary>
    public string? AuthorName { get; }

    /// <summary>
    /// Gets the concatenated text of all <see cref="TextContent"/> parts in this message.
    /// Non-text parts (images, tool calls) are ignored.
    /// </summary>
    public string Text => string.Concat(Contents.OfType<TextContent>().Select(t => t.Text));

    /// <summary>
    /// Gets the tool-call requests contained in this message (typically present on assistant messages
    /// whose finish reason was <see cref="ChatFinishReason.ToolCalls"/>).
    /// </summary>
    public IReadOnlyList<ToolCallContent> ToolCalls => Contents.OfType<ToolCallContent>().ToList();

    /// <summary>
    /// Creates a system message carrying high-level instructions.
    /// </summary>
    /// <param name="text">The system instructions.</param>
    /// <returns>A new system-role message.</returns>
    public static ChatMessage System(string text) => new(ChatRole.System, text);

    /// <summary>
    /// Creates a user message.
    /// </summary>
    /// <param name="text">The user's text.</param>
    /// <returns>A new user-role message.</returns>
    public static ChatMessage User(string text) => new(ChatRole.User, text);

    /// <summary>
    /// Creates an assistant message from text.
    /// </summary>
    /// <param name="text">The assistant's text.</param>
    /// <returns>A new assistant-role message.</returns>
    public static ChatMessage Assistant(string text) => new(ChatRole.Assistant, text);

    /// <summary>
    /// Creates an assistant message from explicit content parts (for example, one or more
    /// <see cref="ToolCallContent"/> parts the model produced).
    /// </summary>
    /// <param name="contents">The content parts.</param>
    /// <returns>A new assistant-role message.</returns>
    public static ChatMessage Assistant(IReadOnlyList<AiContent> contents) => new(ChatRole.Assistant, contents);

    /// <summary>
    /// Creates a tool-result message answering a prior <see cref="ToolCallContent"/>.
    /// </summary>
    /// <param name="callId">The id of the tool call being answered.</param>
    /// <param name="result">The tool's output.</param>
    /// <param name="isError">Whether the tool invocation failed.</param>
    /// <returns>A new tool-role message.</returns>
    public static ChatMessage Tool(string callId, string result, bool isError = false) =>
        new(ChatRole.Tool, new AiContent[] { new ToolResultContent(callId, result, isError) });
}
