using System.Runtime.CompilerServices;
using System.Text;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IChatClient{T}"/> decorator that calls a real inner client and records each request/response
/// into an <see cref="IChatInteractionStore"/>. Pair it with <see cref="ReplayingChatClient{T}"/> to capture a
/// run once and replay it deterministically thereafter.
/// </summary>
/// <typeparam name="T">The numeric type of the inner client.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A tape recorder around the model. It passes your request to the real model and
/// quietly saves the answer keyed by the request, so you can play it back later without spending another call.
/// Streaming calls are recorded too: the streamed pieces are reassembled into the final answer and saved when
/// the stream finishes.
/// </para>
/// </remarks>
public sealed class RecordingChatClient<T> : IChatClient<T>
{
    private readonly IChatClient<T> _inner;
    private readonly IChatInteractionStore _store;

    /// <summary>
    /// Initializes a new recording client.
    /// </summary>
    /// <param name="inner">The real client to call and record.</param>
    /// <param name="store">The store that receives recorded interactions.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="store"/> is <c>null</c>.</exception>
    public RecordingChatClient(IChatClient<T> inner, IChatInteractionStore store)
    {
        Guard.NotNull(inner);
        Guard.NotNull(store);
        _inner = inner;
        _store = store;
    }

    /// <inheritdoc/>
    public string ModelId => _inner.ModelId;

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        // Capture the key BEFORE awaiting: if the caller mutates the message
        // list or options while the request is in flight, the response must
        // still be saved under the key of the request that was actually sent.
        // The inner model's identity keeps recordings isolated when one store
        // is shared across clients for different models.
        var key = ChatInteractionKey.For(messages, options, _inner.ModelId);
        var response = await _inner.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        _store.Save(key, response);
        return response;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var key = ChatInteractionKey.For(messages, options, _inner.ModelId);

        // Buffer the stream into a final ChatResponse so streaming callers are
        // just as replayable as non-streaming ones.
        var textBuilder = new StringBuilder();
        var toolCalls = new SortedDictionary<int, ToolCallAccumulator>();
        var finishReason = ChatFinishReason.Stop;
        ChatUsage? usage = null;

        await foreach (var update in _inner.GetStreamingResponseAsync(messages, options, cancellationToken)
            .ConfigureAwait(false))
        {
            if (update.TextDelta is { } delta)
            {
                textBuilder.Append(delta);
            }

            if (update.ToolCall is { } toolCall)
            {
                if (!toolCalls.TryGetValue(toolCall.Index, out var accumulator))
                {
                    accumulator = new ToolCallAccumulator();
                    toolCalls[toolCall.Index] = accumulator;
                }

                accumulator.CallId ??= toolCall.CallId;
                accumulator.ToolName ??= toolCall.ToolName;
                if (toolCall.ArgumentsJsonFragment is { } fragment)
                {
                    accumulator.Arguments.Append(fragment);
                }
            }

            if (update.FinishReason is { } reason)
            {
                finishReason = reason;
            }

            if (update.Usage is { } reportedUsage)
            {
                usage = reportedUsage;
            }

            yield return update;
        }

        var contents = new List<AiContent>();
        if (textBuilder.Length > 0)
        {
            contents.Add(new TextContent(textBuilder.ToString()));
        }

        foreach (var accumulator in toolCalls.Values)
        {
            if (accumulator.CallId is { } callId && accumulator.ToolName is { } toolName)
            {
                contents.Add(new ToolCallContent(callId, toolName, accumulator.Arguments.ToString()));
            }
        }

        var response = new ChatResponse(
            ChatMessage.Assistant(contents),
            finishReason,
            usage,
            _inner.ModelId);
        _store.Save(key, response);
    }

    private sealed class ToolCallAccumulator
    {
        public string? CallId { get; set; }

        public string? ToolName { get; set; }

        public StringBuilder Arguments { get; } = new();
    }
}
