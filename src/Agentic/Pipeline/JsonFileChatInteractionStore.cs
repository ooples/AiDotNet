using System.Security.Cryptography;
using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Validation;
using Newtonsoft.Json;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>A durable, crash-safe <see cref="IChatInteractionStore"/> backed by one JSON file.</summary>
/// <remarks>
/// <para>
/// This is what turns a model-driven experiment into a repeatable one. Record a run once against a real provider,
/// commit the resulting file, and every later run replays the identical answers with no network access, no API
/// key, and no cost — in a unit test, on a build server, on a reviewer's laptop, a year later. A benchmark whose
/// model calls are recorded can be re-derived by anyone; one whose calls are live cannot, which is precisely the
/// gap between a reproducible result and an anecdote. The reference OpenEvolve implementation logs prompts into
/// its program database for inspection but never replays them, so its runs cannot be reproduced at all.
/// </para>
/// <para>
/// Writes are atomic in the same way <c>JsonEvolutionCheckpointStore</c> is: the whole file is serialized to a
/// temporary file in the same directory, flushed to disk, and swapped into place, keeping the previous version
/// beside it. A crash therefore leaves either the old file or the new one, never a half-written one.
/// </para>
/// <para>
/// Requests are recorded under the SHA-256 of their canonical key, not under the key itself, so the file holds no
/// prompt text, no conversation content, and no credentials — only fingerprints and the answers they map to. Set
/// <c>storeRequestKeys</c> when you are debugging a replay miss and accept that the file will then contain the
/// prompts.
/// </para>
/// <para><b>For Beginners:</b> This saves what the AI said, in a file, so you can play it back later without
/// calling the AI again. Point it at a path, record once, and then switch your client to replay: your tests run
/// instantly, offline, and give the same answers every time. Only fingerprints of your prompts are written, so
/// the file is safe to commit alongside your code.</para>
/// </remarks>
public sealed class JsonFileChatInteractionStore : IChatInteractionStore
{
    /// <summary>The schema version written into every file.</summary>
    public const int SchemaVersion = 1;

    private readonly object _gate = new();
    private readonly Dictionary<string, StoredInteraction> _entries = new(StringComparer.Ordinal);
    private readonly string _path;
    private readonly string _previousPath;
    private readonly long _maxBytes;
    private readonly bool _autoFlush;
    private readonly bool _storeRequestKeys;
    private bool _dirty;

    /// <summary>Initializes a file-backed store, loading any interactions the file already holds.</summary>
    /// <param name="filePath">The JSON file the interactions live in.</param>
    /// <param name="autoFlush">Whether every save writes the file immediately; <c>false</c> requires <see cref="Flush"/>.</param>
    /// <param name="storeRequestKeys">Whether the canonical request key, which contains prompt text, is written too.</param>
    /// <param name="maxBytes">The largest file this store will read or write.</param>
    /// <exception cref="ArgumentNullException"><paramref name="filePath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="filePath"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxBytes"/> is not positive.</exception>
    /// <exception cref="InvalidDataException">The file exists but is not a valid interaction store.</exception>
    public JsonFileChatInteractionStore(
        string filePath,
        bool autoFlush = true,
        bool storeRequestKeys = false,
        long maxBytes = 64L * 1024L * 1024L)
    {
        Guard.NotNullOrWhiteSpace(filePath);
        if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes), maxBytes, "Value must be positive.");

        _path = Path.GetFullPath(filePath);
        _previousPath = _path + ".previous";
        _maxBytes = maxBytes;
        _autoFlush = autoFlush;
        _storeRequestKeys = storeRequestKeys;

        string? directory = Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        Load();
    }

    /// <summary>Gets the full path of the file this store reads and writes.</summary>
    public string FilePath => _path;

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

    /// <summary>Gets whether interactions have been saved that the file does not yet contain.</summary>
    public bool HasUnsavedChanges
    {
        get
        {
            lock (_gate)
            {
                return _dirty;
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
            _entries[Fingerprint(key)] = new StoredInteraction(_storeRequestKeys ? key : null, response);
            _dirty = true;
            if (_autoFlush) Persist();
        }
    }

    /// <inheritdoc/>
    public bool TryGet(string key, out ChatResponse response)
    {
        Guard.NotNullOrWhiteSpace(key);
        lock (_gate)
        {
            if (_entries.TryGetValue(Fingerprint(key), out StoredInteraction? stored))
            {
                response = stored.Response;
                return true;
            }
        }

        response = new ChatResponse(ChatMessage.Assistant(string.Empty));
        return false;
    }

    /// <summary>Writes any pending interactions to the file.</summary>
    /// <exception cref="InvalidDataException">The encoded file would exceed the configured byte limit.</exception>
    public void Flush()
    {
        lock (_gate)
        {
            if (!_dirty) return;
            Persist();
        }
    }

    /// <summary>Removes every interaction from memory and from the file.</summary>
    /// <exception cref="InvalidDataException">The encoded file would exceed the configured byte limit.</exception>
    public void Clear()
    {
        lock (_gate)
        {
            _entries.Clear();
            _dirty = true;
            Persist();
        }
    }

    /// <summary>Returns a description that names the file and the entry count but no recorded content.</summary>
    /// <returns>The path and entry count.</returns>
    public override string ToString() => $"JsonFileChatInteractionStore({_path}, {Count} entries)";

    private void Load()
    {
        if (!File.Exists(_path)) return;

        string json;
        try
        {
            var info = new FileInfo(_path);
            if (info.Length > _maxBytes)
            {
                throw new InvalidDataException(
                    $"The chat interaction store '{_path}' exceeds the {_maxBytes}-byte limit.");
            }

            using var stream = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            json = reader.ReadToEnd();
        }
        catch (IOException exception)
        {
            throw new InvalidDataException($"The chat interaction store '{_path}' could not be read.", exception);
        }

        StoreDocument? document;
        try
        {
            document = JsonConvert.DeserializeObject<StoreDocument>(json);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"The chat interaction store '{_path}' is not valid JSON.", exception);
        }

        if (document is null)
        {
            throw new InvalidDataException($"The chat interaction store '{_path}' is empty.");
        }

        if (document.SchemaVersion != SchemaVersion)
        {
            throw new InvalidDataException(
                $"The chat interaction store '{_path}' has schema version {document.SchemaVersion}; " +
                $"version {SchemaVersion} is expected.");
        }

        foreach (InteractionDocument entry in document.Entries)
        {
            if (string.IsNullOrWhiteSpace(entry.Id))
            {
                throw new InvalidDataException($"The chat interaction store '{_path}' has an entry without an id.");
            }

            _entries[entry.Id] = new StoredInteraction(entry.RequestKey, entry.ToResponse(_path));
        }
    }

    private void Persist()
    {
        var document = new StoreDocument { SchemaVersion = SchemaVersion };
        var ids = new List<string>(_entries.Keys);
        // Ordinal order, so the same set of interactions always serializes to the
        // same bytes and the file is diffable in review.
        ids.Sort(StringComparer.Ordinal);
        foreach (string id in ids)
        {
            document.Entries.Add(InteractionDocument.From(id, _entries[id]));
        }

        string json = JsonConvert.SerializeObject(document, Formatting.Indented);
        byte[] payload = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(json);
        if (payload.LongLength > _maxBytes)
        {
            throw new InvalidDataException(
                $"The chat interaction store would be {payload.LongLength} bytes, above the {_maxBytes}-byte limit.");
        }

        string directory = Path.GetDirectoryName(_path) ?? ".";
        string tempPath = Path.Combine(directory, $".{Path.GetFileName(_path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(payload, 0, payload.Length);
                stream.Flush(flushToDisk: true);
            }

            if (File.Exists(_path))
            {
                File.Replace(tempPath, _path, _previousPath, ignoreMetadataErrors: true);
            }
            else
            {
                File.Move(tempPath, _path);
            }

            _dirty = false;
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); }
                catch (IOException) { }
            }
        }
    }

    private static string Fingerprint(string key)
    {
        using SHA256 sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(key));
        var hex = new StringBuilder(hash.Length * 2);
        foreach (byte value in hash) hex.Append(value.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
        return hex.ToString();
    }

    private sealed class StoredInteraction
    {
        public StoredInteraction(string? requestKey, ChatResponse response)
        {
            RequestKey = requestKey;
            Response = response;
        }

        public string? RequestKey { get; }

        public ChatResponse Response { get; }
    }

    /// <summary>The on-disk shape of the whole store.</summary>
    private sealed class StoreDocument
    {
        /// <summary>Gets or sets the schema version.</summary>
        public int SchemaVersion { get; set; }

        /// <summary>Gets the recorded interactions.</summary>
        public List<InteractionDocument> Entries { get; } = new();
    }

    /// <summary>The on-disk shape of one recorded interaction.</summary>
    private sealed class InteractionDocument
    {
        /// <summary>Gets or sets the SHA-256 fingerprint of the canonical request key.</summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>Gets or sets the canonical request key, present only when key storage was enabled.</summary>
        public string? RequestKey { get; set; }

        /// <summary>Gets or sets the model that produced the response.</summary>
        public string? ModelId { get; set; }

        /// <summary>Gets or sets why generation stopped.</summary>
        public string FinishReason { get; set; } = nameof(ChatFinishReason.Stop);

        /// <summary>Gets or sets the input token count, when it was reported.</summary>
        public int? InputTokens { get; set; }

        /// <summary>Gets or sets the output token count, when it was reported.</summary>
        public int? OutputTokens { get; set; }

        /// <summary>Gets or sets the assistant author name, when one was reported.</summary>
        public string? AuthorName { get; set; }

        /// <summary>Gets the content parts of the assistant message.</summary>
        public List<ContentDocument> Contents { get; } = new();

        public static InteractionDocument From(string id, StoredInteraction stored)
        {
            ChatResponse response = stored.Response;
            var document = new InteractionDocument
            {
                Id = id,
                RequestKey = stored.RequestKey,
                ModelId = response.ModelId,
                FinishReason = response.FinishReason.ToString(),
                InputTokens = response.Usage?.InputTokens,
                OutputTokens = response.Usage?.OutputTokens,
                AuthorName = response.Message.AuthorName
            };

            foreach (AiContent part in response.Message.Contents)
            {
                switch (part)
                {
                    case TextContent text:
                        document.Contents.Add(new ContentDocument { Kind = "text", Text = text.Text });
                        break;
                    case ToolCallContent call:
                        document.Contents.Add(new ContentDocument
                        {
                            Kind = "toolCall",
                            CallId = call.CallId,
                            ToolName = call.ToolName,
                            ArgumentsJson = call.ArgumentsJson
                        });
                        break;
                    case ToolResultContent result:
                        document.Contents.Add(new ContentDocument
                        {
                            Kind = "toolResult",
                            CallId = result.CallId,
                            Text = result.Result,
                            IsError = result.IsError
                        });
                        break;
                    default:
                        // An unrecognized part cannot be replayed faithfully, and a
                        // silently dropped part would make replay differ from the
                        // original response without anyone noticing.
                        throw new InvalidDataException(
                            $"Chat content of type '{part.GetType().Name}' cannot be recorded.");
                }
            }

            return document;
        }

        public ChatResponse ToResponse(string path)
        {
            var contents = new List<AiContent>();
            foreach (ContentDocument part in Contents)
            {
                switch (part.Kind)
                {
                    case "text":
                        contents.Add(new TextContent(part.Text ?? string.Empty));
                        break;
                    case "toolCall":
                        // Bound to locals rather than tested with string.IsNullOrWhiteSpace:
                        // the .NET Framework reference assemblies carry no null-state
                        // attribute on that method, so the compiler cannot see the check
                        // there and the same source would fail only on net471.
                        if (part.CallId is not { } callId || callId.Trim().Length == 0
                            || part.ToolName is not { } toolName || toolName.Trim().Length == 0)
                        {
                            throw new InvalidDataException(
                                $"A recorded tool call in '{path}' is missing its call id or tool name.");
                        }

                        contents.Add(new ToolCallContent(callId, toolName, part.ArgumentsJson));
                        break;
                    case "toolResult":
                        if (part.CallId is not { } resultCallId || resultCallId.Trim().Length == 0)
                        {
                            throw new InvalidDataException($"A recorded tool result in '{path}' is missing its call id.");
                        }

                        contents.Add(new ToolResultContent(resultCallId, part.Text ?? string.Empty, part.IsError));
                        break;
                    default:
                        throw new InvalidDataException(
                            $"The chat interaction store '{path}' has an unknown content kind '{part.Kind}'.");
                }
            }

            if (contents.Count == 0) contents.Add(new TextContent(string.Empty));

#if NET5_0_OR_GREATER
            if (!Enum.TryParse(FinishReason, ignoreCase: true, out ChatFinishReason finishReason))
#else
            if (!TryParseFinishReason(FinishReason, out ChatFinishReason finishReason))
#endif
            {
                throw new InvalidDataException(
                    $"The chat interaction store '{path}' has an unknown finish reason '{FinishReason}'.");
            }

            ChatUsage? usage = InputTokens.HasValue && OutputTokens.HasValue
                ? new ChatUsage(InputTokens.Value, OutputTokens.Value)
                : null;

            return new ChatResponse(
                new ChatMessage(ChatRole.Assistant, contents, AuthorName),
                finishReason,
                usage,
                ModelId);
        }

#if !NET5_0_OR_GREATER
        private static bool TryParseFinishReason(string text, out ChatFinishReason reason)
        {
            foreach (ChatFinishReason candidate in (ChatFinishReason[])Enum.GetValues(typeof(ChatFinishReason)))
            {
                if (string.Equals(candidate.ToString(), text, StringComparison.OrdinalIgnoreCase))
                {
                    reason = candidate;
                    return true;
                }
            }

            reason = ChatFinishReason.Stop;
            return false;
        }
#endif
    }

    /// <summary>The on-disk shape of one content part.</summary>
    private sealed class ContentDocument
    {
        /// <summary>Gets or sets the part kind: <c>text</c>, <c>toolCall</c>, or <c>toolResult</c>.</summary>
        public string Kind { get; set; } = "text";

        /// <summary>Gets or sets the text of a text part or the payload of a tool result.</summary>
        public string? Text { get; set; }

        /// <summary>Gets or sets the tool-call identifier.</summary>
        public string? CallId { get; set; }

        /// <summary>Gets or sets the tool name of a tool call.</summary>
        public string? ToolName { get; set; }

        /// <summary>Gets or sets the serialized arguments of a tool call.</summary>
        public string? ArgumentsJson { get; set; }

        /// <summary>Gets or sets whether a tool result reported an error.</summary>
        public bool IsError { get; set; }
    }
}
