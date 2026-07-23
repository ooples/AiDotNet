using System.Collections.Concurrent;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// An <see cref="IConversationStore"/> that persists thread histories to a single JSON file, so
/// conversations survive process restarts without an external database. Each message is stored as its role
/// plus its text (the durable dialogue); multimodal and tool-call parts are not persisted.
/// </summary>
/// <remarks>
/// <para>
/// All threads live in one file as a <c>{ threadId: [ {role, text}, ... ] }</c> map. Reads and writes are
/// serialized with an in-process lock and the whole file is rewritten on each append, which suits modest
/// single-process workloads. For concurrent or high-volume scenarios, use a database-backed store.
/// </para>
/// <para><b>For Beginners:</b> The same notebook as the in-memory store, but written to a file on disk, so
/// closing and reopening your app keeps the conversation history.
/// </para>
/// </remarks>
public sealed class JsonFileConversationStore : IConversationStore
{
    private static readonly JsonSerializerSettings SerializerSettings = new()
    {
        Formatting = Formatting.Indented,
        Converters = { new StringEnumConverter() }
    };

    // One gate per canonical file path, shared process-wide: every operation
    // is a load-modify-rewrite of the whole file, so two store instances
    // pointed at the same path must serialize against EACH OTHER — an
    // instance-scoped lock would let the last writer silently drop the other
    // instance's update. Case-insensitive keying over-locks (never
    // under-locks) on case-sensitive file systems, which is the safe side.
    private static readonly ConcurrentDictionary<string, object> Gates =
        new(StringComparer.OrdinalIgnoreCase);

    private readonly object _gate;
    private readonly string _filePath;

    /// <summary>
    /// Initializes a store backed by the given file. The file is created on first write; a missing file is
    /// treated as an empty store.
    /// </summary>
    /// <param name="filePath">The path of the backing JSON file. Must be non-empty.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="filePath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is empty/whitespace.</exception>
    public JsonFileConversationStore(string filePath)
    {
        Guard.NotNullOrWhiteSpace(filePath);
        // Canonicalise so the stored path is absolute and free of "../"
        // segments — defends against relative-path traversal surprises later
        // even though we intentionally don't restrict to a base directory
        // (callers may legitimately persist conversations anywhere they have
        // write access).
        _filePath = Path.GetFullPath(filePath);
        _gate = Gates.GetOrAdd(_filePath, _ => new object());
    }

    /// <inheritdoc/>
    public Task AppendAsync(string threadId, IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);
        Guard.NotNull(messages);

        lock (_gate)
        {
            var store = Load();
            if (!store.TryGetValue(threadId, out var history))
            {
                history = new List<StoredMessage>();
                store[threadId] = history;
            }

            foreach (var message in messages)
            {
                Guard.NotNull(message);
                history.Add(new StoredMessage { Role = message.Role, Text = message.Text });
            }

            Save(store);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ChatMessage>> GetAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);

        lock (_gate)
        {
            var store = Load();
            IReadOnlyList<ChatMessage> result = store.TryGetValue(threadId, out var history)
                ? history.Select(m => new ChatMessage(m.Role, m.Text)).ToList()
                : new List<ChatMessage>();
            return Task.FromResult(result);
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<string>> ListThreadsAsync(CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            IReadOnlyList<string> ids = new List<string>(Load().Keys);
            return Task.FromResult(ids);
        }
    }

    /// <inheritdoc/>
    public Task ClearAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);

        lock (_gate)
        {
            var store = Load();
            if (store.Remove(threadId))
            {
                Save(store);
            }
        }

        return Task.CompletedTask;
    }

    private Dictionary<string, List<StoredMessage>> Load()
    {
        if (!File.Exists(_filePath))
        {
            return new Dictionary<string, List<StoredMessage>>(StringComparer.Ordinal);
        }

        var json = File.ReadAllText(_filePath);
        if (json.Trim().Length == 0)
        {
            return new Dictionary<string, List<StoredMessage>>(StringComparer.Ordinal);
        }

        try
        {
            var loaded = JsonConvert.DeserializeObject<Dictionary<string, List<StoredMessage>>>(json, SerializerSettings);
            return loaded ?? new Dictionary<string, List<StoredMessage>>(StringComparer.Ordinal);
        }
        catch (JsonException ex)
        {
            // Rethrow with the file path so callers can recover or surface
            // a useful diagnostic instead of an opaque JSON parse error.
            throw new InvalidOperationException(
                $"Failed to deserialise conversation store at '{_filePath}'. " +
                "The backing file may be corrupted or hand-edited.", ex);
        }
    }

    private void Save(Dictionary<string, List<StoredMessage>> store)
    {
        // Write-then-rename for crash safety: a power loss / kill mid-write
        // leaves the original _filePath intact instead of a half-flushed file
        // and corrupted history. File.Move with overwrite is atomic on NTFS
        // and POSIX file systems, which covers our supported targets.
        var json = JsonConvert.SerializeObject(store, SerializerSettings);
        var tempPath = _filePath + ".tmp";
        File.WriteAllText(tempPath, json);
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        File.Move(tempPath, _filePath, overwrite: true);
#else
        if (File.Exists(_filePath))
        {
            File.Replace(tempPath, _filePath, destinationBackupFileName: null);
        }
        else
        {
            File.Move(tempPath, _filePath);
        }
#endif
    }

    private sealed class StoredMessage
    {
        public ChatRole Role { get; set; }

        public string Text { get; set; } = string.Empty;
    }
}
