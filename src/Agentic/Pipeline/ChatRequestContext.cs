using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// The mutable request state flowing through a chat-middleware pipeline. Middleware can inspect and rewrite
/// the <see cref="Messages"/> and <see cref="Options"/> before the model is called, and share state via
/// <see cref="Items"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Think of this as the request envelope passed down an assembly line. Each
/// station (middleware) can read it, change it (add a system instruction, tweak settings), or stash a note in
/// the bag for later stations, before it reaches the model.
/// </para>
/// </remarks>
public sealed class ChatRequestContext
{
    /// <summary>
    /// Initializes a new request context.
    /// </summary>
    /// <param name="messages">The conversation to send. Must be non-null.</param>
    /// <param name="options">Per-call options, or <c>null</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    public ChatRequestContext(IReadOnlyList<ChatMessage> messages, ChatOptions? options)
    {
        Guard.NotNull(messages);
        Messages = messages;
        Options = options;
    }

    /// <summary>Gets or sets the conversation to send (middleware may rewrite it).</summary>
    public IReadOnlyList<ChatMessage> Messages { get; set; }

    /// <summary>Gets or sets the per-call options (middleware may rewrite them).</summary>
    public ChatOptions? Options { get; set; }

    /// <summary>Gets a property bag for sharing state between middleware stages.</summary>
    public IDictionary<string, object?> Items { get; } = new Dictionary<string, object?>(StringComparer.Ordinal);
}
