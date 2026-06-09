using System.Text;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// Wraps any <see cref="IAgent{T}"/> with long-term memory recall: before each run it searches an
/// <see cref="IAgentMemoryStore"/> for memories relevant to the latest user message and injects them as
/// context, so the agent answers with knowledge gathered across previous conversations.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// This is retrieval-augmented <em>memory</em> (RAG over the agent's own remembered facts). It composes with
/// the rest of the stack: the inner agent may be an <see cref="AgentExecutor{T}"/>, a
/// <see cref="SupervisorAgent{T}"/>, or a <see cref="Swarm{T}"/>, and the result can in turn be wrapped by a
/// <see cref="ThreadedAgent{T}"/> for short-term conversation memory. Writing memories is intentionally
/// explicit (call <see cref="IAgentMemoryStore.AddAsync"/>) — this wrapper only reads, so what gets
/// remembered stays under the application's control.
/// </para>
/// <para><b>For Beginners:</b> Before answering, the assistant flips through its long-term notes, pulls out
/// the ones related to your question, and reads them first — so it can use things it learned in earlier,
/// separate chats. It doesn't decide on its own what to write down; your app does that.
/// </para>
/// </remarks>
public sealed class MemoryAugmentedAgent<T> : IAgent<T>
{
    private readonly IAgent<T> _inner;
    private readonly IAgentMemoryStore _memory;
    private readonly MemoryAugmentationOptions _options;

    /// <summary>
    /// Initializes a new memory-augmented wrapper.
    /// </summary>
    /// <param name="inner">The agent to give long-term recall to.</param>
    /// <param name="memory">The long-term memory store to search.</param>
    /// <param name="options">Recall settings. <c>null</c> uses defaults.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="memory"/> is <c>null</c>.</exception>
    public MemoryAugmentedAgent(IAgent<T> inner, IAgentMemoryStore memory, MemoryAugmentationOptions? options = null)
    {
        Guard.NotNull(inner);
        Guard.NotNull(memory);
        _inner = inner;
        _memory = memory;
        _options = options ?? new MemoryAugmentationOptions();
    }

    /// <inheritdoc/>
    public string Name => _inner.Name;

    /// <inheritdoc/>
    public string Description => _inner.Description;

    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="messages"/> is empty.</exception>
    public async Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("The conversation must contain at least one message.", nameof(messages));
        }

        var query = LatestUserText(messages);
        if (query is null || query.Trim().Length == 0)
        {
            return await _inner.RunAsync(messages, cancellationToken).ConfigureAwait(false);
        }

        var topK = _options.TopK is { } configured && configured > 0
            ? configured
            : MemoryAugmentationOptions.DefaultTopK;

        var matches = await _memory.SearchAsync(query, topK, cancellationToken).ConfigureAwait(false);
        var relevant = _options.MinScore is { } minScore
            ? matches.Where(m => m.Score >= minScore).ToList()
            : matches.ToList();

        if (relevant.Count == 0)
        {
            return await _inner.RunAsync(messages, cancellationToken).ConfigureAwait(false);
        }

        var contextMessage = ChatMessage.System(BuildMemoryBlock(relevant));
        var augmented = new List<ChatMessage>(messages.Count + 1) { contextMessage };
        augmented.AddRange(messages);

        return await _inner.RunAsync(augmented, cancellationToken).ConfigureAwait(false);
    }

    private string BuildMemoryBlock(IReadOnlyList<ScoredMemory> memories)
    {
        var header = _options.Header is { } configured && configured.Trim().Length > 0
            ? configured
            : MemoryAugmentationOptions.DefaultHeader;

        var builder = new StringBuilder();
        builder.AppendLine(header);
        foreach (var scored in memories)
        {
            builder.Append("- ").AppendLine(scored.Memory.Content);
        }

        return builder.ToString().TrimEnd();
    }

    private static string? LatestUserText(IReadOnlyList<ChatMessage> messages)
    {
        for (var i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == ChatRole.User)
            {
                return messages[i].Text;
            }
        }

        return null;
    }
}
