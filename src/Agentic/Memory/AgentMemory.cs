namespace AiDotNet.Agentic.Memory;

/// <summary>
/// A single long-term memory: a piece of text the agent should be able to recall later, with a stable id
/// and optional metadata. Memories live across conversation threads (unlike <see cref="IConversationStore"/>,
/// which is per-thread short-term history).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Think of one sticky note the assistant keeps — e.g. "the user prefers metric
/// units" or "the project deadline is in June". Each note has a unique id (so it can be updated or removed)
/// and the note text itself. Later, when something relevant comes up, the assistant can find and re-read the
/// note even if it's in a completely different conversation.
/// </para>
/// </remarks>
public sealed class AgentMemory
{
    /// <summary>
    /// Initializes a new memory.
    /// </summary>
    /// <param name="id">The stable, unique identifier for this memory.</param>
    /// <param name="content">The remembered text. Must be non-empty.</param>
    /// <param name="metadata">Optional key/value metadata (e.g., source, category). <c>null</c> means none.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="id"/> or <paramref name="content"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="id"/> or <paramref name="content"/> is empty/whitespace.</exception>
    public AgentMemory(string id, string content, IReadOnlyDictionary<string, string>? metadata = null)
    {
        Guard.NotNullOrWhiteSpace(id);
        Guard.NotNullOrWhiteSpace(content);
        Id = id;
        Content = content;
        Metadata = metadata;
    }

    /// <summary>Gets the stable, unique identifier for this memory.</summary>
    public string Id { get; }

    /// <summary>Gets the remembered text.</summary>
    public string Content { get; }

    /// <summary>Gets optional key/value metadata, or <c>null</c> when none was supplied.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; }
}
