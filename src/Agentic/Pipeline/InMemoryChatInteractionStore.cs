using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// A process-local <see cref="IChatInteractionStore"/> holding recorded chat interactions in memory. Ideal
/// for tests and within-process record/replay; contents are lost when the process exits.
/// </summary>
public sealed class InMemoryChatInteractionStore : IChatInteractionStore
{
    private readonly object _gate = new();
    private readonly Dictionary<string, ChatResponse> _entries = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _entries.Count;
            }
        }
    }

    /// <inheritdoc/>
    public void Save(string key, ChatResponse response)
    {
        Guard.NotNullOrWhiteSpace(key);
        Guard.NotNull(response);
        lock (_gate)
        {
            _entries[key] = response;
        }
    }

    /// <inheritdoc/>
    public bool TryGet(string key, out ChatResponse response)
    {
        Guard.NotNullOrWhiteSpace(key);
        lock (_gate)
        {
            if (_entries.TryGetValue(key, out var found))
            {
                response = found;
                return true;
            }
        }

        response = new ChatResponse(ChatMessage.Assistant(string.Empty));
        return false;
    }
}
