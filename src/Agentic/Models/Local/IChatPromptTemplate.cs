using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Renders a chat conversation (a list of <see cref="ChatMessage"/>) into the single prompt string a local
/// language model is fed. Different model families expect different role markers, so this is pluggable.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Cloud chat APIs accept a list of messages directly, but a raw local model
/// just continues one block of text. This converts the conversation into that block, tagging who said what
/// (system/user/assistant) in the format the model was trained on, and ends with the assistant's turn so the
/// model knows it should reply next.
/// </para>
/// </remarks>
public interface IChatPromptTemplate
{
    /// <summary>
    /// Renders the conversation into a prompt string.
    /// </summary>
    /// <param name="messages">The conversation so far. Must be non-empty.</param>
    /// <returns>The prompt text to encode and feed to the model.</returns>
    string Render(IReadOnlyList<ChatMessage> messages);
}
