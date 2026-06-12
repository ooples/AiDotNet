using System.Text;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A ChatML-style <see cref="IChatPromptTemplate"/> that renders each message as
/// <c>&lt;|role|&gt;\n{text}\n</c> and ends with an open <c>&lt;|assistant|&gt;</c> turn for the model to
/// complete. This is a widely-used, model-agnostic default.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This writes the conversation like a script: each line is tagged with who is
/// speaking (<c>&lt;|user|&gt;</c>, <c>&lt;|assistant|&gt;</c>, ...), and the script stops right where the
/// assistant is about to speak — so the model fills in the assistant's reply.
/// </para>
/// </remarks>
internal sealed class ChatMlPromptTemplate : IChatPromptTemplate
{
    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="messages"/> is empty.</exception>
    public string Render(IReadOnlyList<ChatMessage> messages)
    {
        Guard.NotNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("The conversation must contain at least one message.", nameof(messages));
        }

        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            builder.Append("<|").Append(RoleTag(message.Role)).Append("|>\n");
            builder.Append(message.Text).Append('\n');
        }

        builder.Append("<|assistant|>\n");
        return builder.ToString();
    }

    private static string RoleTag(ChatRole role) => role switch
    {
        ChatRole.System => "system",
        ChatRole.User => "user",
        ChatRole.Assistant => "assistant",
        ChatRole.Tool => "tool",
        // A new ChatRole must force an explicit template decision — silently
        // remapping it to "user" would rewrite prompt semantics.
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, "Unsupported chat role for the ChatML template."),
    };
}
