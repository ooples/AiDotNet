namespace AiDotNet.Agentic.Models;

/// <summary>
/// Base type for a single piece of content inside a <see cref="ChatMessage"/>.
/// </summary>
/// <remarks>
/// <para>
/// A chat message is not just a string — it is a list of content parts. This lets one message mix
/// modalities and structured items: plain text, images, a request from the model to call a tool
/// (<see cref="ToolCallContent"/>), or the result of running a tool (<see cref="ToolResultContent"/>).
/// Concrete subclasses are matched with pattern matching, e.g. <c>part is TextContent t</c>.
/// </para>
/// <para><b>For Beginners:</b> Older APIs treated a message as one block of text. Real conversations
/// are richer — a single turn might contain a sentence <em>and</em> an image, or the model might
/// reply with "please run the calculator tool" instead of text. Representing a message as a list of
/// typed parts lets us model all of that cleanly. Each part is one of the subclasses of this class.
/// </para>
/// </remarks>
public abstract class AiContent
{
    /// <summary>
    /// Initializes the base content part. Protected because <see cref="AiContent"/> is abstract and
    /// only constructed through a concrete subclass.
    /// </summary>
    private protected AiContent()
    {
    }
}
