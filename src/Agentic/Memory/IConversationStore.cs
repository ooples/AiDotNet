using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// Persists multi-turn conversations keyed by a thread id, so an agent can remember earlier turns across
/// separate runs. This is the short-term ("thread") memory the agent stack composes via
/// <see cref="ThreadedAgent{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// A store keeps an ordered list of dialogue messages per thread (typically user and assistant turns).
/// Implementations range from in-process (<see cref="InMemoryConversationStore"/>) to durable
/// (<see cref="JsonFileConversationStore"/>), with the same contract, so callers swap persistence without
/// changing agent code. Implementations persist a message's role and text; multimodal parts and tool-call
/// metadata are not part of the durable dialogue.
/// </para>
/// <para><b>For Beginners:</b> This is the notebook where a chat's history is written down under a label
/// (the thread id). Next time the same conversation continues, the agent reads the notebook first so it
/// remembers what was already said.
/// </para>
/// </remarks>
public interface IConversationStore
{
    /// <summary>
    /// Appends messages to the end of a thread's history, creating the thread if it does not exist.
    /// </summary>
    /// <param name="threadId">The conversation thread id. Must be non-empty.</param>
    /// <param name="messages">The messages to append (in order). Must be non-null.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task AppendAsync(string threadId, IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the ordered message history for a thread, or an empty list when the thread is unknown.
    /// </summary>
    /// <param name="threadId">The conversation thread id. Must be non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The thread's messages in order (empty when none).</returns>
    Task<IReadOnlyList<ChatMessage>> GetAsync(string threadId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists the ids of all known threads.
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The known thread ids (order unspecified).</returns>
    Task<IReadOnlyList<string>> ListThreadsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Removes a thread and its history. A no-op when the thread is unknown.
    /// </summary>
    /// <param name="threadId">The conversation thread id. Must be non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task ClearAsync(string threadId, CancellationToken cancellationToken = default);
}
