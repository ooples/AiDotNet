namespace AiDotNet.Agentic.Models;

/// <summary>
/// Identifies who authored a <see cref="ChatMessage"/> in a conversation.
/// </summary>
/// <remarks>
/// <para>
/// Modern chat models are message-based rather than prompt-based: instead of one big string,
/// a request is a list of messages, each tagged with the role of its author. The role tells the
/// model how to treat the content (instructions, user input, its own prior replies, or tool output).
/// </para>
/// <para><b>For Beginners:</b> Think of a chat as a transcript of a conversation. Every line in the
/// transcript has a speaker. This enum is the list of possible speakers:
/// - <b>System</b>: the setup instructions ("You are a helpful assistant that answers in French").
/// - <b>User</b>: the human's messages.
/// - <b>Assistant</b>: the model's own replies.
/// - <b>Tool</b>: the result of running a tool/function the model asked to call.
///
/// Using a fixed set of roles (an enum) instead of free-text strings means a typo like "asistant"
/// can't slip through and break a request at runtime.
/// </para>
/// </remarks>
public enum ChatRole
{
    /// <summary>
    /// High-level instructions that steer the model's behavior for the whole conversation.
    /// Usually the first message. Some providers call this the "developer" role.
    /// </summary>
    System,

    /// <summary>
    /// Input authored by the end user (the human asking questions or giving instructions).
    /// </summary>
    User,

    /// <summary>
    /// Output authored by the model itself. Prior assistant messages are replayed back to the model
    /// so it remembers what it already said.
    /// </summary>
    Assistant,

    /// <summary>
    /// The result produced by executing a tool/function that the assistant requested.
    /// A tool message is correlated to the originating call via its tool-call id.
    /// </summary>
    Tool
}
