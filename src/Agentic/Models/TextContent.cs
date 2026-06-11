namespace AiDotNet.Agentic.Models;

/// <summary>
/// A plain-text content part within a <see cref="ChatMessage"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the most common kind of content — ordinary words. Most messages
/// you send or receive are a single <see cref="TextContent"/>.
/// </para>
/// </remarks>
public sealed class TextContent : AiContent
{
    /// <summary>
    /// Initializes a new <see cref="TextContent"/> with the given text.
    /// </summary>
    /// <param name="text">The text. May be empty, but not <c>null</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is <c>null</c>.</exception>
    public TextContent(string text)
    {
        Guard.NotNull(text);
        Text = text;
    }

    /// <summary>
    /// Gets the text of this content part.
    /// </summary>
    public string Text { get; }

    /// <inheritdoc/>
    public override string ToString() => Text;
}
