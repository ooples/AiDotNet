using System.Globalization;
using System.Runtime.CompilerServices;
using AiDotNet.Configuration;
using AiDotNet.Validation;
using Newtonsoft.Json;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>An <see cref="IChatClient{T}"/> that asks a person, by writing each prompt to a queue directory.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Every request becomes a JSON file in a directory, and the client waits for a matching answer file to appear
/// beside it. A person, a script, or a different tool writes that answer. It is how a whole evolution run can be
/// driven by hand: useful for building a golden trajectory to replay later, for reviewing exactly what the search
/// asks a model at each step, and for running an evolution against a model that has no API at all.
/// </para>
/// <para>
/// One request writes <c>&lt;id&gt;.task.json</c> holding the conversation and the per-call settings, and waits for
/// <c>&lt;id&gt;.answer.json</c> holding <c>{"text": "..."}</c>. Plain text in the answer file is accepted too, so a
/// person can reply without writing JSON. Both files are removed once the answer is read, so the directory shows
/// only what is still outstanding. Waiting is bounded by <see cref="ManualChatClientOptions.Timeout"/> when one is
/// set and otherwise continues until the request is cancelled, which is the reference implementation's behaviour.
/// </para>
/// <para>
/// A stale answer from a previous run would be served instantly and silently as if it answered the new prompt, so
/// the constructor clears the queue directory unless asked not to.
/// </para>
/// <para><b>For Beginners:</b> Point this at an empty folder and start a run. A file appears there containing the
/// question; write your reply into a file with the same name but ending <c>.answer.json</c>, and the run continues.
/// This is the "I am the model" mode.</para>
/// </remarks>
public sealed class ManualChatClient<T> : IChatClient<T>
{
    /// <summary>The extension of the file this client writes for each request.</summary>
    public const string TaskExtension = ".task.json";

    /// <summary>The extension of the file this client waits for.</summary>
    public const string AnswerExtension = ".answer.json";

    private readonly ManualChatClientOptions _options;
    private long _requests;

    /// <summary>Initializes a client over a queue directory, creating it when it does not exist.</summary>
    /// <param name="queueDirectory">The directory task and answer files are exchanged in.</param>
    /// <param name="options">Polling, timeout, and identity; <c>null</c> uses the defaults.</param>
    /// <exception cref="ArgumentNullException"><paramref name="queueDirectory"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="queueDirectory"/> is blank, or an option is invalid.</exception>
    public ManualChatClient(string queueDirectory, ManualChatClientOptions? options = null)
    {
        Guard.NotNullOrWhiteSpace(queueDirectory);
        ManualChatClientOptions effective = options is null ? new ManualChatClientOptions() : options.Clone();
        effective.Validate();
        _options = effective;

        QueueDirectory = Path.GetFullPath(queueDirectory);
        Directory.CreateDirectory(QueueDirectory);
        if (effective.ClearStaleTasks) ClearQueue();
    }

    /// <summary>Gets the directory task and answer files are exchanged in.</summary>
    public string QueueDirectory { get; }

    /// <inheritdoc/>
    public string ModelId => _options.ModelId;

    /// <summary>Gets how many requests this client has written.</summary>
    public long Requests => Interlocked.Read(ref _requests);

    /// <inheritdoc/>
    /// <exception cref="TimeoutException">No answer appeared within the configured timeout.</exception>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        cancellationToken.ThrowIfCancellationRequested();

        long ordinal = Interlocked.Increment(ref _requests);
        string id = ordinal.ToString("D6", CultureInfo.InvariantCulture) + "-" + Guid.NewGuid().ToString("N");
        string taskPath = Path.Combine(QueueDirectory, id + TaskExtension);
        string answerPath = Path.Combine(QueueDirectory, id + AnswerExtension);

        // Written through a temporary file and moved into place, so whoever is watching the directory never opens a
        // half-written question. It is the same hazard this client guards against on the answer side.
        WriteAtomic(taskPath, JsonConvert.SerializeObject(new TaskDocument
        {
            Id = id,
            ModelId = ModelId,
            RequestedUtc = DateTimeOffset.UtcNow,
            AnswerFileName = Path.GetFileName(answerPath),
            Temperature = options?.Temperature,
            MaxOutputTokens = options?.MaxOutputTokens,
            Seed = options?.Seed,
            Messages = messages.Select(message => new MessageDocument
            {
                Role = message.Role.ToString(),
                Text = message.Text
            }).ToList()
        }, Formatting.Indented));

        try
        {
            string answer = await WaitForAnswerAsync(answerPath, cancellationToken).ConfigureAwait(false);
            return new ChatResponse(new ChatMessage(ChatRole.Assistant, answer), ChatFinishReason.Stop,
                usage: null, modelId: ModelId);
        }
        finally
        {
            TryDelete(taskPath);
            TryDelete(answerPath);
        }
    }

    /// <inheritdoc/>
    /// <remarks>A person answers all at once, so the whole reply arrives as a single update.</remarks>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ChatResponse response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        yield return new ChatResponseUpdate(ChatRole.Assistant, response.Text, finishReason: ChatFinishReason.Stop);
    }

    /// <summary>Removes every task and answer file left behind by an earlier run.</summary>
    /// <returns>How many files were removed.</returns>
    /// <remarks>
    /// A stale answer would be served instantly and silently as the reply to a prompt it never saw, which is worse
    /// than no answer at all: the run continues and its results are quietly meaningless.
    /// </remarks>
    public int ClearQueue()
    {
        int removed = 0;
        foreach (string path in Directory.EnumerateFiles(QueueDirectory))
        {
            if (!path.EndsWith(TaskExtension, StringComparison.OrdinalIgnoreCase) &&
                !path.EndsWith(AnswerExtension, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            if (TryDelete(path)) removed++;
        }
        return removed;
    }

    /// <summary>Waits for the answer file, reading it as JSON when it is JSON and as plain text otherwise.</summary>
    private async Task<string> WaitForAnswerAsync(string answerPath, CancellationToken cancellationToken)
    {
        DateTimeOffset deadline = _options.Timeout.HasValue
            ? DateTimeOffset.UtcNow + _options.Timeout.Value
            : DateTimeOffset.MaxValue;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (File.Exists(answerPath))
            {
                // A writer that is still flushing leaves a file that exists but reads as empty, as unreadable, or as
                // half a JSON document. All three mean "not ready yet" and none of them mean "the answer is this".
                // Returning a truncated document as the reply is the failure this loop exists to prevent.
                string? content = TryReadAll(answerPath);
                if (content is not null && content.Trim().Length > 0 &&
                    TryReadAnswer(content, out string answer))
                {
                    return answer;
                }
            }

            if (DateTimeOffset.UtcNow >= deadline)
            {
                throw new TimeoutException(
                    "No answer file appeared at '" + answerPath + "' within " +
                    _options.Timeout.GetValueOrDefault().ToString() + ".");
            }

            await Task.Delay(_options.PollInterval, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Extracts the reply text from an answer file's contents, if the file is complete.</summary>
    /// <param name="content">The file's contents.</param>
    /// <param name="answer">The reply text when the file is complete.</param>
    /// <returns><c>false</c> when the file looks like a JSON document that is not finished being written.</returns>
    /// <remarks>
    /// Content that starts with a brace is a JSON answer, and one that fails to parse is a file still being written,
    /// not a person's reply. Returning it verbatim would hand a truncated document to the caller as the model's
    /// answer, which then fails to parse as a diff for a reason nothing explains. Content that does not start with a
    /// brace is plain text and is taken as written, because somebody answering by hand should not need JSON.
    /// </remarks>
    private static bool TryReadAnswer(string content, out string answer)
    {
        answer = content;
        string trimmed = content.TrimStart();
        if (trimmed.Length == 0 || trimmed[0] != '{') return true;

        try
        {
            AnswerDocument? document = JsonConvert.DeserializeObject<AnswerDocument>(content);
            if (document is null) return false;
            answer = document.Text ?? string.Empty;
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>Writes a file by creating a temporary one beside it and moving it into place.</summary>
    private static void WriteAtomic(string path, string content)
    {
        string temporary = path + ".tmp";
        File.WriteAllText(temporary, content);
        if (File.Exists(path)) File.Delete(path);
        File.Move(temporary, path);
    }

    private static string? TryReadAll(string path)
    {
        try
        {
            return File.ReadAllText(path);
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static bool TryDelete(string path)
    {
        try
        {
            if (!File.Exists(path)) return false;
            File.Delete(path);
            return true;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>Serialization shape of one queued request.</summary>
    private sealed class TaskDocument
    {
        public string Id { get; set; } = string.Empty;
        public string ModelId { get; set; } = string.Empty;
        public DateTimeOffset RequestedUtc { get; set; }
        public string AnswerFileName { get; set; } = string.Empty;
        public double? Temperature { get; set; }
        public int? MaxOutputTokens { get; set; }
        public long? Seed { get; set; }
        public List<MessageDocument> Messages { get; set; } = new();
    }

    /// <summary>Serialization shape of one conversation message.</summary>
    private sealed class MessageDocument
    {
        public string Role { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
    }

    /// <summary>Serialization shape of an answer file.</summary>
    private sealed class AnswerDocument
    {
        public string? Text { get; set; }
    }
}
