using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// A process-local <see cref="IConversationStore"/> that keeps thread histories in memory. Ideal for tests,
/// single-process apps, and the default zero-config experience; histories are lost when the process exits.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The simplest notebook — kept in RAM. Fast and needs no setup, but forgotten
/// when the program stops. For history that survives restarts, use <see cref="JsonFileConversationStore"/>
/// (or a database-backed store).
/// </para>
/// </remarks>
public sealed class InMemoryConversationStore : IConversationStore
{
    private readonly object _gate = new();
    private readonly Dictionary<string, List<ChatMessage>> _threads = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public Task AppendAsync(string threadId, IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);
        Guard.NotNull(messages);
        cancellationToken.ThrowIfCancellationRequested();

        lock (_gate)
        {
            if (!_threads.TryGetValue(threadId, out var history))
            {
                history = new List<ChatMessage>();
                _threads[threadId] = history;
            }

            foreach (var message in messages)
            {
                Guard.NotNull(message);
                history.Add(message);
            }
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ChatMessage>> GetAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);
        cancellationToken.ThrowIfCancellationRequested();

        lock (_gate)
        {
            IReadOnlyList<ChatMessage> snapshot = _threads.TryGetValue(threadId, out var history)
                ? new List<ChatMessage>(history)
                : new List<ChatMessage>();
            return Task.FromResult(snapshot);
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<string>> ListThreadsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            IReadOnlyList<string> ids = new List<string>(_threads.Keys);
            return Task.FromResult(ids);
        }
    }

    /// <inheritdoc/>
    public Task ClearAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);
        cancellationToken.ThrowIfCancellationRequested();

        lock (_gate)
        {
            _threads.Remove(threadId);
        }

        return Task.CompletedTask;
    }
}
